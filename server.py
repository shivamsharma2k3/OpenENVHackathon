"""Source entry point used by Docker and local `python server.py` runs."""

from app_runtime import app, main


if __name__ == "__main__":
    main()
