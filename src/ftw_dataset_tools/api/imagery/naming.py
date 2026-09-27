"""File names and STAC keys for a chip's per-source season imagery."""

from __future__ import annotations

import re
from typing import Literal, NamedTuple

from ftw_dataset_tools.api.imagery.sources import DEFAULT_SOURCE, SOURCES, source_class

__all__ = [
    "SEASONS",
    "ChildRef",
    "child_item_id",
    "image_filename",
    "imagery_asset_keys",
    "link_source",
    "parent_asset_key",
    "parse_child_id",
    "parse_image_stem",
]

Season = Literal["planting", "harvest"]
SEASONS: tuple[Season, ...] = ("planting", "harvest")

_SOURCE_BY_SUFFIX = {cls.suffix: name for name, cls in SOURCES.items()}
_SUFFIXES = "|".join(_SOURCE_BY_SUFFIX)
_CHILD_RE = re.compile(rf"^(?P<chip>.+)_(?P<season>planting|harvest)_(?P<suffix>{_SUFFIXES})$")
_IMAGE_RE = re.compile(
    rf"^(?P<chip>.+)_(?P<season>planting|harvest)_image_(?P<suffix>{_SUFFIXES})$"
)


class ChildRef(NamedTuple):
    chip_id: str
    season: Season
    source: str


def child_item_id(chip_id: str, season: str, source: str = DEFAULT_SOURCE) -> str:
    return f"{chip_id}_{season}_{source_class(source).suffix}"


def image_filename(chip_id: str, season: str, source: str = DEFAULT_SOURCE) -> str:
    return f"{chip_id}_{season}_image_{source_class(source).suffix}.tif"


def _parse(pattern: re.Pattern[str], value: str) -> ChildRef | None:
    match = pattern.match(value)
    if match is None:
        return None
    return ChildRef(match["chip"], match["season"], _SOURCE_BY_SUFFIX[match["suffix"]])  # type: ignore[arg-type]


def parse_child_id(item_id: str) -> ChildRef | None:
    """Split ``{chip}_{season}_{suffix}`` into its parts, or None for a chip item."""
    return _parse(_CHILD_RE, item_id)


def parse_image_stem(stem: str) -> ChildRef | None:
    """Split ``{chip}_{season}_image_{suffix}`` into its parts."""
    return _parse(_IMAGE_RE, stem)


def parent_asset_key(season: str, kind: str, source: str = DEFAULT_SOURCE) -> str:
    """Parent chip asset key; the default source keeps the unsuffixed name."""
    key = f"{season}_{kind}"
    return key if source == DEFAULT_SOURCE else f"{key}_{source_class(source).suffix}"


def imagery_asset_keys(source: str | None = None) -> tuple[str, ...]:
    """Parent asset keys that downloaded imagery adds, for one source or all of them."""
    names = SOURCES if source is None else (source,)
    keys = [parent_asset_key(season, "image", name) for name in names for season in SEASONS]
    if source in (None, DEFAULT_SOURCE):
        keys.append("thumbnail")
    return tuple(keys)


def link_source(link: object) -> str:
    """The source of an ``ftw:<season>`` link; links written before sources existed are S2."""
    extra = getattr(link, "extra_fields", None) or {}
    return extra.get("ftw:source", DEFAULT_SOURCE)
