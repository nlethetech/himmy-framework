"""Enable ``python -m himmy`` as an alias for the ``himmy`` console script."""

from __future__ import annotations

from himmy.cli.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
