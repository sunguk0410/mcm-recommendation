import io
import logging
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from uuid import uuid4

from PIL import Image, UnidentifiedImageError


logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 10
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


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
    logger.info("Background removal started. imageUrl=%s", image_url)

    try:
        image_data = download_image(image_url)
        image = load_image(image_data)

        try:
            if remover is None:
                from rembg import remove as remover

            result = remover(image)
            if not isinstance(result, Image.Image):
                result = Image.open(io.BytesIO(result))
            result = result.convert("RGBA")
            result.load()
        except Exception as error:
            raise BackgroundRemovalError(
                f"Background removal processing failed: {error}"
            ) from error

        temporary_path = None
        try:
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
        except (OSError, ValueError) as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise ImageSaveError(f"Unable to save processed image: {error}") from error

        logger.info(
            "Background removal completed. imageUrl=%s, filename=%s",
            image_url,
            filename,
        )
        return filename
    except Exception as error:
        logger.exception(
            "Background removal failed. imageUrl=%s, filename=%s, error=%s",
            image_url,
            filename,
            error,
        )
        raise
