"""Compatibility wrapper for the packaged server entry point."""

from server.app import app, main


if __name__ == "__main__":
    main()
