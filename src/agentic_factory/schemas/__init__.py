"""Pydantic schemas used by runtime agents."""

from pydantic import BaseModel

from agentic_factory.schemas.brand_screener import BrandScreenerResult
from agentic_factory.schemas.data_runner import DataRunnerResult
from agentic_factory.schemas.keepa_downloader import KeepaDownloadResult


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "BrandScreenerResult": BrandScreenerResult,
    "DataRunnerResult": DataRunnerResult,
    "KeepaDownloadResult": KeepaDownloadResult,
}


def get_schema(name: str) -> type[BaseModel]:
    """Return a registered output schema by name."""
    try:
        return SCHEMA_REGISTRY[name]
    except KeyError as exc:
        available = list(SCHEMA_REGISTRY.keys())
        raise ValueError(f"Unknown schema: {name}. Available: {available}") from exc


__all__ = [
    "BrandScreenerResult",
    "DataRunnerResult",
    "KeepaDownloadResult",
    "SCHEMA_REGISTRY",
    "get_schema",
]
