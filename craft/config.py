"""Runtime configuration for the standalone CRAFT pipeline.

Replaces the Django settings that the pipeline used inside the Insilico Sigma
platform. Defaults reproduce the platform's behaviour exactly:

  CRAFT_CACHE_DIR        root of the API response caches.
                         Default: <package>/data/cache, the same layout the
                         platform used (BASE_DIR/craft/data/cache).
  CRAFT_CACHE_TTL_HOURS  cache time-to-live in hours. Default 168 (7 days),
                         as on the platform. Set to 0 to make cache entries
                         never expire: required to reproduce the published
                         results offline from the shipped cache snapshot,
                         whose entries are older than seven days.
  CRAFT_DISABLE_IUPHAR   1 = do not use IUPHAR/GtoPdb for any target (the
                         setting used for the published results; see
                         services/iuphar_client.py).
"""
import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def cache_root() -> Path:
    return Path(os.environ.get("CRAFT_CACHE_DIR") or (_PACKAGE_DIR / "data" / "cache"))


def cache_dir(name: str) -> Path:
    """Cache directory for one source (chembl, uniprot, iuphar, pdbe, ...)."""
    return cache_root() / name


def cache_ttl_hours(default: int = 168) -> int:
    """TTL in hours; 0 in the environment means 'never expire'."""
    raw = os.environ.get("CRAFT_CACHE_TTL_HOURS", "").strip()
    if raw == "":
        return default
    hours = int(raw)
    return 10 ** 9 if hours == 0 else hours
