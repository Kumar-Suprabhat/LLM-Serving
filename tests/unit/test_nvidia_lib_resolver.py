"""Tests for :func:`llmopt.vllm_launcher.resolve_nvidia_library_path`.

The resolver is load-bearing: without it, vLLM 0.25.x's startup import of ``torchcodec``
fails to ``dlopen libnvrtc.so.13`` and the engine never reaches ``/health``. Every launch
depends on the string this function returns being correct, so the shape of its output and
its behaviour on missing directories are pinned here rather than discovered on the host.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from llmopt.vllm_launcher import resolve_nvidia_library_path


def _make_site_packages(venv: Path) -> Path:
    """Create the ``lib/python3.12/site-packages`` skeleton inside a tmp venv and return it."""
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    return site_packages


def _make_nvidia_lib(site_packages: Path, package: str) -> Path:
    """Create ``<site-packages>/nvidia/<package>/lib`` and return it."""
    directory = site_packages / "nvidia" / package / "lib"
    directory.mkdir(parents=True)
    return directory


def test_returns_both_paths_when_both_exist(tmp_path: Path) -> None:
    """Both NVIDIA runtimes are useful; the resolver must emit them in loader-search order."""
    site_packages = _make_site_packages(tmp_path)
    cu13_lib = _make_nvidia_lib(site_packages, "cu13")
    nvrtc_lib = _make_nvidia_lib(site_packages, "cuda_nvrtc")

    result = resolve_nvidia_library_path(PurePosixPath(str(tmp_path)))

    assert result == f"{cu13_lib}:{nvrtc_lib}"


def test_returns_only_cu13_when_cuda_nvrtc_missing(tmp_path: Path) -> None:
    """``cu13`` alone carries ``libnvrtc.so.13``; missing ``cuda_nvrtc`` is not fatal."""
    site_packages = _make_site_packages(tmp_path)
    cu13_lib = _make_nvidia_lib(site_packages, "cu13")

    result = resolve_nvidia_library_path(PurePosixPath(str(tmp_path)))

    assert result == str(cu13_lib)


def test_returns_only_cuda_nvrtc_when_cu13_missing(tmp_path: Path) -> None:
    """A venv with only ``cuda_nvrtc`` is a legitimate state; emit just that directory."""
    site_packages = _make_site_packages(tmp_path)
    nvrtc_lib = _make_nvidia_lib(site_packages, "cuda_nvrtc")

    result = resolve_nvidia_library_path(PurePosixPath(str(tmp_path)))

    assert result == str(nvrtc_lib)


def test_returns_empty_string_when_no_nvidia_dirs(tmp_path: Path) -> None:
    """A CPU-only venv has neither directory; the resolver returns ``""`` rather than raising.

    An exception here would make every CPU-side unit test need a GPU-shaped fixture, which
    would defeat the point of having pure helpers.
    """
    _make_site_packages(tmp_path)

    result = resolve_nvidia_library_path(PurePosixPath(str(tmp_path)))

    assert result == ""


def test_handles_missing_site_packages(tmp_path: Path) -> None:
    """A venv skeleton without ``lib/python*/site-packages`` must not raise.

    The resolver is called before anyone has verified the venv is intact, so a half-built
    tree -- or a non-existent path -- is normal input, not an error condition.
    """
    result = resolve_nvidia_library_path(PurePosixPath(str(tmp_path / "no-venv-here")))

    assert result == ""


def test_returns_absolute_path_strings(tmp_path: Path) -> None:
    """``LD_LIBRARY_PATH`` is read by the dynamic loader, which does not resolve relatives.

    Every directory the resolver emits must be absolute, or the loader will silently search
    relative to the engine's working directory instead of the venv. The test asserts the
    rendered string embeds the tmp root rather than splitting on ``:``, because on Windows
    the drive-letter colon would mangle a naive split; the real deployment runs on Linux
    where the two paths join cleanly.
    """
    site_packages = _make_site_packages(tmp_path)
    cu13_lib = _make_nvidia_lib(site_packages, "cu13")
    nvrtc_lib = _make_nvidia_lib(site_packages, "cuda_nvrtc")

    result = resolve_nvidia_library_path(PurePosixPath(str(tmp_path)))

    assert str(cu13_lib) in result
    assert str(nvrtc_lib) in result
    assert Path(str(cu13_lib)).is_absolute()
    assert Path(str(nvrtc_lib)).is_absolute()
