"""Allow running the package via ``python -m plain_mlops``."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

