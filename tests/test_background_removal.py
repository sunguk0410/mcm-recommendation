import io

import pytest
from PIL import Image

from src import background_removal


def png_bytes(mode="RGB", size=(8, 8)):
    buffer = io.BytesIO()
    Image.new(mode, size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_remove_background_saves_rgba_png(monkeypatch, tmp_path):
    monkeypatch.setattr(
        background_removal,
        "download_image",
        lambda _url: png_bytes(),
    )

    filename = background_removal.remove_background(
        "https://example.com/avatar.png",
        tmp_path,
        remover=lambda image: image,
    )

    assert filename.startswith("avatar-no-bg-")
    assert filename.endswith(".png")
    with Image.open(tmp_path / filename) as result:
        assert result.format == "PNG"
        assert result.mode == "RGBA"


def test_load_image_rejects_unsupported_format():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="BMP")

    with pytest.raises(background_removal.UnsupportedImageError):
        background_removal.load_image(buffer.getvalue())


def test_load_image_rejects_excessive_pixel_count(monkeypatch):
    monkeypatch.setattr(background_removal, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(background_removal.ImageTooLargeError):
        background_removal.load_image(png_bytes(size=(4, 4)))


def test_processing_failure_has_specific_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        background_removal,
        "download_image",
        lambda _url: png_bytes(),
    )

    def fail(_image):
        raise RuntimeError("model failed")

    with pytest.raises(background_removal.BackgroundRemovalError):
        background_removal.remove_background(
            "https://example.com/avatar.png",
            tmp_path,
            remover=fail,
        )


def test_rembg_session_is_created_once_and_reused(monkeypatch):
    created_sessions = []

    def fake_new_session(model_name):
        session = object()
        created_sessions.append((model_name, session))
        return session

    monkeypatch.setattr(background_removal, "_rembg_session", None)
    monkeypatch.setattr(
        background_removal,
        "import_rembg",
        lambda: (object(), fake_new_session),
    )

    first_session, first_reused = background_removal.get_rembg_session()
    second_session, second_reused = background_removal.get_rembg_session()

    assert first_session is second_session
    assert first_reused is False
    assert second_reused is True
    assert created_sessions == [(background_removal.REMBG_MODEL_NAME, first_session)]


def test_startup_initialization_runs_inference_warmup(monkeypatch):
    session = object()
    warmup_calls = []

    def fake_remove(image, session):
        warmup_calls.append((image.size, image.mode, session))
        return image

    monkeypatch.setattr(
        background_removal,
        "import_rembg",
        lambda: (fake_remove, object()),
    )
    monkeypatch.setattr(
        background_removal,
        "get_rembg_session",
        lambda: (session, False),
    )

    background_removal.initialize_background_removal()

    assert warmup_calls == [((320, 320), "RGB", session)]
