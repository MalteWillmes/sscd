"""Finding and converting input images."""

import numpy as np
import pytest
from PIL import Image

from sscd_libs.data_processing import list_input_images, to_8bit_rgb
from sscd_libs.settings import input_dir_allowed, load_settings


def _touch_image(path):
    Image.new("RGB", (8, 8)).save(path)


def test_image_types_any_case_and_folder_with_brackets(tmp_path):
    folder = tmp_path / "scales [2021]"
    folder.mkdir()
    for name in ("a.tif", "b.TIF", "c.tiff", "d.JPG", "e.jpeg"):
        _touch_image(folder / name)
    (folder / "notes.txt").write_text("not an image")
    found = [p.split("\\")[-1].split("/")[-1] for p in list_input_images(folder)]
    assert found == ["a.tif", "b.TIF", "c.tiff", "d.JPG", "e.jpeg"]


def test_no_images_and_clashing_names_are_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_input_images(tmp_path)
    _touch_image(tmp_path / "a.tif")
    _touch_image(tmp_path / "a.jpg")
    with pytest.raises(ValueError, match="same file name"):
        list_input_images(tmp_path)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA", "P"])
def test_common_modes_become_8bit_rgb(mode):
    assert to_8bit_rgb(Image.new(mode, (4, 4))).mode == "RGB"


def test_16bit_is_scaled_not_clipped():
    grey16 = np.array([[0, 257 * 128, 65535]], dtype=np.uint16)
    out = np.asarray(to_8bit_rgb(Image.fromarray(grey16)))
    assert out[0, :, 0].tolist() == [0, 128, 255]


def test_allowed_input_roots(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.toml"
    settings_file.write_text(f'allowed_input_roots = ["{(tmp_path / "archive").as_posix()}"]\n')
    monkeypatch.setenv("SSCD_SETTINGS", str(settings_file))
    settings = load_settings()
    assert input_dir_allowed(tmp_path / "archive" / "2024", settings)
    assert not input_dir_allowed(tmp_path / "elsewhere", settings)
    assert not input_dir_allowed(tmp_path / "archive" / ".." / "elsewhere", settings)


def test_duplicate_images_are_found(tmp_path):
    from sscd_libs.data_processing import find_duplicate_images

    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    a.save(tmp_path / "a.tif")
    a.save(tmp_path / "a (1).tif")         # identical copy
    b.save(tmp_path / "b.tif")              # same size, different content
    groups = find_duplicate_images(sorted(str(p) for p in tmp_path.glob("*.tif")))
    assert [[p.split("\\")[-1].split("/")[-1] for p in g] for g in groups] == [["a (1).tif", "a.tif"]]
