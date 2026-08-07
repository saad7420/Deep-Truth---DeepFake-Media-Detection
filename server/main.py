import app.utils.env  # noqa

import uvicorn

from app.core import create_app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info",
    )