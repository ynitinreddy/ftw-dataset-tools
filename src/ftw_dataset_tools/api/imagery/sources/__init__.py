"""Imagery sources that scene selection and download can draw from."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ftw_dataset_tools.api.imagery.sources.base import (
    ChipAssessment,
    FetchResult,
    ImagerySource,
    ImagerySourceError,
    SearchResult,
    SourceUnavailableError,
)
from ftw_dataset_tools.api.imagery.sources.planetscope import (
    DEFAULT_BUNDLE,
    PLANET_BUNDLES,
    PlanetScopeSource,
)
from ftw_dataset_tools.api.imagery.sources.sentinel2 import SEARCH_BACKENDS, Sentinel2Source

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_BUNDLE",
    "DEFAULT_SOURCE",
    "PLANET_BUNDLES",
    "SEARCH_BACKENDS",
    "SOURCES",
    "SOURCE_NAMES",
    "ChipAssessment",
    "FetchResult",
    "ImagerySource",
    "ImagerySourceError",
    "PlanetScopeSource",
    "SearchResult",
    "Sentinel2Source",
    "SourceUnavailableError",
    "build_source",
    "source_class",
]

SOURCES: dict[str, type[ImagerySource]] = {
    Sentinel2Source.name: Sentinel2Source,
    PlanetScopeSource.name: PlanetScopeSource,
}
SOURCE_NAMES = tuple(SOURCES)
DEFAULT_SOURCE = Sentinel2Source.name


def source_class(name: str) -> type[ImagerySource]:
    """The source class registered under ``name``."""
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"Unknown imagery source {name!r}; use one of {list(SOURCES)}.") from None


def build_source(
    name: str = DEFAULT_SOURCE,
    *,
    s2_collection: str = "c1",
    search_backend: str = "parquet",
    planet_bundle: str = DEFAULT_BUNDLE,
    planet_harmonize: bool = True,
    planet_wait: bool = True,
    planet_timeout_minutes: float = 60.0,
) -> ImagerySource:
    """A configured source; options for other sources are ignored."""
    builders: dict[str, Callable[[], ImagerySource]] = {
        Sentinel2Source.name: lambda: Sentinel2Source(
            collection=s2_collection, backend=search_backend
        ),
        PlanetScopeSource.name: lambda: PlanetScopeSource(
            bundle=planet_bundle,
            harmonize=planet_harmonize,
            wait=planet_wait,
            timeout_minutes=planet_timeout_minutes,
        ),
    }
    source_class(name)
    return builders[name]()
