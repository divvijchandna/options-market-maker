"""Smoke test: the package is installed and importable."""

import mm


def test_version_is_set_from_installed_metadata() -> None:
    # "+unknown" means the fallback fired, i.e. the package is not installed.
    assert isinstance(mm.__version__, str)
    assert mm.__version__
    assert "+unknown" not in mm.__version__
