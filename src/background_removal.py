import io
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from uuid import uuid4

from PIL import Image, UnidentifiedImageError


# Uvicorn only configures its own loggers at INFO level. Using a child of
# ``uvicorn.error`` makes the timing records visible in container logs while
# keeping the application's existing log format.
logger = logging.getLogger("uvicorn.error.background_removal")

DOWNLOAD_TIMEOUT_SECONDS = 10
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
REMBG_MODEL_NAME = os.getenv("REMBG_MODEL_NAME", "u2net")

_rembg_session = None
_rembg_remove = None
_rembg_new_session = None
_rembg_import_lock = threading.Lock()
_rembg_session_lock = threading.Lock()
_request_count = 0
_request_count_lock = threading.Lock()


class ImageDownloadError(Exception):
    pass


class ImageTooLargeError(Exception):
    pass


class UnsupportedImageError(Exception):
    pass


class BackgroundRemovalError(Exception):
    pass


class ImageSaveError(Exception):
    pass


def import_rembg():
    """Import rembg once, outside the latency-sensitive request path."""
    global _rembg_remove, _rembg_new_session
    if _rembg_remove is not None and _rembg_new_session is not None:
        return _rembg_remove, _rembg_new_session

    with _rembg_import_lock:
        if _rembg_remove is None or _rembg_new_session is None:
            started_at = time.perf_counter()
            logger.info("rembg import started")
            from rembg import new_session, remove

            _rembg_remove = remove
            _rembg_new_session = new_session
            logger.info(
                "rembg import completed. elapsedMs=%.1f",
                (time.perf_counter() - started_at) * 1000,
            )
    return _rembg_remove, _rembg_new_session


def get_rembg_session():
    """Create the ONNX session once and reuse it across requests."""
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session, True

    _, new_session = import_rembg()

    with _rembg_session_lock:
        if _rembg_session is None:
            started_at = time.perf_counter()
            logger.info("rembg session initialization started. model=%s", REMBG_MODEL_NAME)
            _rembg_session = new_session(REMBG_MODEL_NAME)
            logger.info(
                "rembg session initialization completed. model=%s, elapsedMs=%.1f",
                REMBG_MODEL_NAME,
                (time.perf_counter() - started_at) * 1000,
            )
            return _rembg_session, False
    return _rembg_session, True


def initialize_background_removal():
    """Load rembg and run one inference during application startup."""
    started_at = time.perf_counter()
    logger.info("Background removal startup initialization started")
    remover, _ = import_rembg()
    session, session_reused = get_rembg_session()

    warmup_started_at = time.perf_counter()
    logger.info(
        "Background removal inference warmup started. model=%s",
        REMBG_MODEL_NAME,
    )
    warmup_image = Image.new("RGB", (320, 320), "white")
    warmup_result = remover(warmup_image, session=session)
    if isinstance(warmup_result, Image.Image):
        warmup_result.load()
    else:
        with Image.open(io.BytesIO(warmup_result)) as warmup_output:
            warmup_output.load()
    logger.info(
        "Background removal inference warmup completed. model=%s, elapsedMs=%.1f",
        REMBG_MODEL_NAME,
        (time.perf_counter() - warmup_started_at) * 1000,
    )

    logger.info(
        "Background removal startup initialization completed. "
        "model=%s, sessionReused=%s, elapsedMs=%.1f",
        REMBG_MODEL_NAME,
        session_reused,
        (time.perf_counter() - started_at) * 1000,
    )


def _next_request_number():
    global _request_count
    with _request_count_lock:
        _request_count += 1
        return _request_count


def download_image(
    image_url: str,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> bytes:
    request = urllib.request.Request(
        image_url,
        headers={"User-Agent": "mcm-recommendation-background-removal/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise ImageTooLargeError(
                            f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit"
                        )
                except ValueError:
                    pass

            chunks = []
            downloaded = 0
            while True:
                chunk = response.read(min(64 * 1024, max_bytes + 1 - downloaded))
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ImageTooLargeError(
                        f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit"
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    except ImageTooLargeError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise ImageDownloadError(f"Unable to download image: {error}") from error


def load_image(image_data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.format not in SUPPORTED_IMAGE_FORMATS:
            raise UnsupportedImageError(
                f"Unsupported image format: {image.format or 'unknown'}"
            )
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ImageTooLargeError(
                f"Image exceeds the {MAX_IMAGE_PIXELS:,} pixel limit"
            )
        image.load()
        return image.convert("RGBA")
    except (UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise UnsupportedImageError("Unsupported or invalid image format") from error
    except (OSError, ValueError) as error:
        raise UnsupportedImageError("Unsupported or invalid image format") from error


def remove_background(
    image_url: str,
    output_directory: Path,
    remover: Callable[[Image.Image], Image.Image] | None = None,
) -> str:
    filename = f"avatar-no-bg-{uuid4()}.png"
    request_number = _next_request_number()
    request_started_at = time.perf_counter()
    logger.info(
        "Background removal started. requestNumber=%s, imageUrl=%s",
        request_number,
        image_url,
    )

    try:
        step_started_at = time.perf_counter()
        image_data = download_image(image_url)
        logger.info(
            "Background source downloaded. requestNumber=%s, imageUrl=%s, "
            "bytes=%s, elapsedMs=%.1f",
            request_number,
            image_url,
            len(image_data),
            (time.perf_counter() - step_started_at) * 1000,
        )

        step_started_at = time.perf_counter()
        image = load_image(image_data)
        logger.info(
            "Background source decoded. requestNumber=%s, imageUrl=%s, "
            "width=%s, height=%s, elapsedMs=%.1f",
            request_number,
            image_url,
            image.width,
            image.height,
            (time.perf_counter() - step_started_at) * 1000,
        )

        try:
            step_started_at = time.perf_counter()
            if remover is None:
                remover, _ = import_rembg()
                session_acquire_started_at = time.perf_counter()
                session, session_reused = get_rembg_session()
                logger.info(
                    "Background removal session acquired. requestNumber=%s, "
                    "model=%s, sessionReused=%s, elapsedMs=%.1f",
                    request_number,
                    REMBG_MODEL_NAME,
                    session_reused,
                    (time.perf_counter() - session_acquire_started_at) * 1000,
                )
                step_started_at = time.perf_counter()
                result = remover(image, session=session)
            else:
                step_started_at = time.perf_counter()
                result = remover(image)
            if not isinstance(result, Image.Image):
                result = Image.open(io.BytesIO(result))
            result = result.convert("RGBA")
            result.load()
            logger.info(
                "Background removal inference completed. requestNumber=%s, "
                "imageUrl=%s, elapsedMs=%.1f",
                request_number,
                image_url,
                (time.perf_counter() - step_started_at) * 1000,
            )
        except Exception as error:
            raise BackgroundRemovalError(
                f"Background removal processing failed: {error}"
            ) from error

        temporary_path = None
        try:
            step_started_at = time.perf_counter()
            output_directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=output_directory,
                prefix=".avatar-no-bg-",
                suffix=".png",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                result.save(temporary_file, format="PNG")
            os.replace(temporary_path, output_directory / filename)
            logger.info(
                "Background removal image saved. requestNumber=%s, filename=%s, "
                "elapsedMs=%.1f",
                request_number,
                filename,
                (time.perf_counter() - step_started_at) * 1000,
            )
        except (OSError, ValueError) as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise ImageSaveError(f"Unable to save processed image: {error}") from error

        logger.info(
            "Background removal completed. requestNumber=%s, imageUrl=%s, "
            "filename=%s, elapsedMs=%.1f",
            request_number,
            image_url,
            filename,
            (time.perf_counter() - request_started_at) * 1000,
        )
        return filename
    except Exception as error:
        logger.exception(
            "Background removal failed. requestNumber=%s, imageUrl=%s, filename=%s, "
            "elapsedMs=%.1f, error=%s",
            request_number,
            image_url,
            filename,
            (time.perf_counter() - request_started_at) * 1000,
            error,
        )
        raise
