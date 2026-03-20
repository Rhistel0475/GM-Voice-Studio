"""
Thin entrypoint: run the FastAPI app from app.main.
Usage: python server.py  (or: python -m app.main with uvicorn in __main__)
"""
import os as _os
import socket
import sys
try:
    from dotenv import load_dotenv
    load_dotenv(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from app.main import app
from app.core.config import SERVER_NAME, PORT
from app.api.routers.health import health, ready

__all__ = ["app", "health", "ready"]


def _can_bind_port(host: str, port: int) -> bool:
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        test_socket.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        test_socket.close()

if __name__ == "__main__":
    import uvicorn
    bind_host = SERVER_NAME if SERVER_NAME != "0.0.0.0" else "127.0.0.1"
    if not _can_bind_port(bind_host, PORT):
        print(
            f"ERROR: Port {PORT} is already in use. Stop the existing backend process and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    uvicorn.run(app, host=SERVER_NAME, port=PORT)
