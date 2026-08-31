"""FastAPI application for the SPhyR Environment.

This module creates an HTTP server that exposes the SPhyREnvironment over the
OpenEnv HTTP and WebSocket endpoints.

Run locally::

    uvicorn server.app:app --host 0.0.0.0 --port 8000
    python -m server.app --port 8000
"""

import os

from openenv.core.env_server.http_server import create_app

try:
    from models import SPhyRAction, SPhyRObservation
except ImportError:
    from ..models import SPhyRAction, SPhyRObservation

from .sphyr_environment import SPhyREnvironment

_max_concurrent_raw = os.environ.get("MAX_CONCURRENT_ENVS", "8")
max_concurrent = int(_max_concurrent_raw)

# The dataset a served environment starts on. Clients override it per episode
# with reset(subject=...).
DEFAULT_SUBJECT = os.environ.get("SPHYR_SUBJECT", "1_random_cell_easy")


def create_sphyr_environment() -> SPhyREnvironment:
    """Factory: a fresh environment per WebSocket session.

    Each session gets its own RNG and sample cursor, so concurrent clients do
    not interleave each other's sample order.
    """
    return SPhyREnvironment(subject=DEFAULT_SUBJECT)


app = create_app(
    create_sphyr_environment,
    SPhyRAction,
    SPhyRObservation,
    env_name="sphyr",
    max_concurrent_envs=max_concurrent,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the SPhyR environment server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    main(port=args.port)
