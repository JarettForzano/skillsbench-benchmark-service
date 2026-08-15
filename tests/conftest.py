import os

from pytest import Config


def pytest_configure(config: Config) -> None:
    """Configure pytest with the auth-disabled test profile."""
    os.environ["AUTH_DISABLED"] = "true"
