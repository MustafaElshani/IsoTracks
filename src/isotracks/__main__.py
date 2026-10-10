"""Allow ``python -m isotracks`` alongside the ``isotracks`` console script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
