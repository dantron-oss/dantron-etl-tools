"""``python -m`` with this package's name runs the command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
