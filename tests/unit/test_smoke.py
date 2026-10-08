"""Smoke test: the package is installed and importable."""

import savia


def test_package_is_importable() -> None:
    assert savia.__name__ == "savia"
    savia.main()
