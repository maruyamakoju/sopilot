from __future__ import annotations

import os

import uvicorn

from .api import create_app


app = create_app()


def run() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("sopilot.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
