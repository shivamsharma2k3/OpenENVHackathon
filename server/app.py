"""Packaged server entry point required for multi-mode deployment."""

from app_runtime import app, main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
