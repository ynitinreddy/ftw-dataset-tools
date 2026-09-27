"""The interface an imagery source implements for scene selection and download."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    import pystac

__all__ = [
    "ChipAssessment",
    "FetchResult",
    "ImagerySource",
    "ImagerySourceError",
    "SearchResult",
    "SourceUnavailableError",
    "search_window",
    "short_date",
]


def short_date(item: pystac.Item) -> str:
    """Month-day of an item's acquisition for log lines (e.g. ``6-05``)."""
    if item.datetime:
        return f"{item.datetime.month}-{item.datetime.day:02d}"
    return item.id[:15]


def search_window(center_date: datetime, buffer_days: int) -> tuple[datetime, datetime]:
    """First and last instant of the days within ``buffer_days`` of ``center_date``."""
    start = (center_date - timedelta(days=buffer_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = (center_date + timedelta(days=buffer_days)).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    return start, end


class ImagerySourceError(Exception):
    """An imagery source could not search, assess or fetch a scene."""


class SourceUnavailableError(ImagerySourceError):
    """A source's optional dependency or credentials are missing."""


@dataclass(frozen=True)
class SearchResult:
    """Candidate scenes for one search window, best scene-level cloud cover first."""

    items: list[pystac.Item]
    description: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChipAssessment:
    """Cloud and nodata measured over one chip's footprint."""

    cloud_cover: float
    cloud_cover_source: Literal["pixel", "scene"]
    cloud_mask: str | None = None
    nodata: float | None = None


@dataclass
class FetchResult:
    """Where to read each band from, or why that is not possible yet."""

    status: Literal["ready", "pending", "failed"]
    bands: dict[str, tuple[str, int]] = field(default_factory=dict)
    error: str | None = None
    cleanup: list[Path] = field(default_factory=list)


class ImagerySource(Protocol):
    """A satellite imagery source: searches scenes, rates them per chip, fetches pixels."""

    name: ClassVar[str]
    suffix: ClassVar[str]
    title: ClassVar[str]
    default_workers: ClassVar[int]
    reflectance_bands: ClassVar[frozenset[str]]

    @property
    def stac_host(self) -> str: ...

    def search(
        self, bbox: tuple[float, float, float, float], center_date: datetime, buffer_days: int
    ) -> SearchResult: ...

    def scene_cloud_cover(self, item: pystac.Item) -> float: ...

    def assess(
        self,
        item: pystac.Item,
        bbox: tuple[float, float, float, float],
        nodata_max: float,
        log: Callable[[str], None],
    ) -> ChipAssessment | None: ...

    def child_assets(self, item: pystac.Item) -> dict[str, pystac.Asset]: ...

    def child_properties(self, item: pystac.Item) -> dict: ...

    def fetch(
        self,
        child: pystac.Item,
        child_path: Path | None,
        bands: list[str],
        log: Callable[[str], None],
    ) -> FetchResult: ...
