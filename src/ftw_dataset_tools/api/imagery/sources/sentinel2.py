"""Sentinel-2 L2A from the Earth Search STAC API or its STAC-GeoParquet mirror."""

from __future__ import annotations

import re
import threading
import time
import urllib.error
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, ClassVar

import pystac_client
from pystac.extensions.eo import EOExtension

from ftw_dataset_tools.api.imagery import parquet_search
from ftw_dataset_tools.api.imagery.cloud_analysis import calculate_pixel_cloud_cover
from ftw_dataset_tools.api.imagery.nodata_analysis import (
    calculate_nodata_percentage,
    get_nodata_from_metadata,
)
from ftw_dataset_tools.api.imagery.settings import (
    CHILD_ITEM_BAND_ASSETS,
    CHILD_ITEM_BANDS,
    DEFAULT_CLOUD_COVER_SCENE,
    PIXEL_CHECK_SKIP_THRESHOLD,
    REFLECTANCE_BANDS,
    S2_COLLECTIONS,
    STAC_URL,
)
from ftw_dataset_tools.api.imagery.sources.base import (
    ChipAssessment,
    FetchResult,
    SearchResult,
    search_window,
    short_date,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    import pystac

__all__ = ["SEARCH_BACKENDS", "STACQueryResult", "Sentinel2Source"]

SEARCH_BACKENDS = ("parquet", "earth-search")


def _format_date_range(center_date: datetime, buffer_days: int) -> str:
    """Format date range for STAC API queries."""
    start = (center_date - timedelta(days=buffer_days)).strftime("%Y-%m-%dT00:00:00Z")
    end = (center_date + timedelta(days=buffer_days)).strftime("%Y-%m-%dT23:59:59Z")
    return f"{start}/{end}"


# One client per thread. A client owns an HTTP session and a plain dict of
# resolved STAC objects, neither of which is built for concurrent use, and
# selection runs several chips at once - so caching per thread keeps the reuse
# without the sharing.
_CLIENTS = threading.local()


def _get_stac_client(catalog_url: str) -> pystac_client.Client:
    """Get this thread's STAC client for a catalog URL, opening it on first use."""
    clients: dict[str, pystac_client.Client] | None = getattr(_CLIENTS, "by_url", None)
    if clients is None:
        clients = {}
        _CLIENTS.by_url = clients

    client = clients.get(catalog_url)
    if client is None:
        client = pystac_client.Client.open(catalog_url)
        clients[catalog_url] = client

    return client


@dataclass
class STACQueryResult:
    """Result of a STAC query with debug information."""

    items: list[pystac.Item]
    catalog_url: str
    collection: str
    bbox: tuple[float, float, float, float]
    date_range: str
    cloud_cover_max: int


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (502, 503, 504)
    return isinstance(error, (urllib.error.URLError, OSError))


def _query_stac(
    bbox: tuple[float, float, float, float],
    center_date: datetime,
    cloud_cover_max: int,
    buffer_days: int,
    s2_collection: str = "c1",
    max_retries: int = 3,
) -> STACQueryResult:
    """Query Earth Search for Sentinel-2 scenes, retrying transient errors."""
    date_range = _format_date_range(center_date, buffer_days)
    collection = S2_COLLECTIONS.get(s2_collection, "sentinel-2-c1-l2a")

    for attempt in range(max_retries):
        try:
            search = _get_stac_client(STAC_URL).search(
                collections=[collection],
                bbox=list(bbox),
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": cloud_cover_max}},
            )
            items = list(search.items())
            break
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            if not _is_retryable(e) or attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)

    items.sort(key=lambda item: EOExtension.ext(item).cloud_cover or 100)
    return STACQueryResult(
        items=items,
        catalog_url=STAC_URL,
        collection=collection,
        bbox=bbox,
        date_range=date_range,
        cloud_cover_max=cloud_cover_max,
    )


def _query_parquet(
    bbox: tuple[float, float, float, float],
    center_date: datetime,
    cloud_cover_max: int,
    buffer_days: int,
    s2_collection: str = "c1",
) -> STACQueryResult:
    """Run the :func:`_query_stac` search against the rate-limit-free parquet mirror."""
    start, end = search_window(center_date, buffer_days)
    collection = S2_COLLECTIONS.get(s2_collection, parquet_search.COLLECTION_C1)
    items = parquet_search.query_scenes(
        bbox=bbox, start=start, end=end, cloud_cover_max=cloud_cover_max, collection=collection
    )
    return STACQueryResult(
        items=items,
        catalog_url=f"{parquet_search.DEFAULT_MIRROR_ROOT}/{collection}",
        collection=collection,
        bbox=bbox,
        date_range=_format_date_range(center_date, buffer_days),
        cloud_cover_max=cloud_cover_max,
    )


def _missing_bands_error(item: pystac.Item, bands: list[str]) -> str:
    """Explain why an item carries no asset for any of the requested `bands`.

    A completed download replaces the child's band assets with a single local
    `image` (or adds `clipped` under --keep-remote-refs), so one cause of an empty
    band set is that this scene is already on disk - a symptom worth naming, since
    "no matching band assets" reads like a broken catalog.

    That only holds for bands a child item ever carries. A request for, say,
    swir16 fails on an untouched catalog too, so blaming the local asset there
    would be a false diagnosis - and would hide the list of what is available.
    """
    local_asset = next((key for key in ("image", "clipped") if key in item.assets), None)
    never_carried = [band for band in bands if band not in CHILD_ITEM_BAND_ASSETS]
    if local_asset is not None and not never_carried:
        return (
            "Scene is already downloaded: its band assets were replaced by the local "
            f"'{local_asset}' asset, leaving nothing to fetch. Resume (the default) skips "
            "these; to re-download, run select-images --force first to restore the remote "
            "band refs (plain select-images skips chips that already have a selection)."
        )
    return f"No matching band assets found in scene. Available: {list(item.assets.keys())}"


def _chip_nodata(
    item: pystac.Item,
    bbox: tuple[float, float, float, float],
    nodata_max: float,
    log: Callable[[str], None],
) -> tuple[bool, float | None]:
    """Whether to reject the scene for nodata in the chip window, and the % measured.

    A scene-level value of zero proves the chip's window is clean; any other value
    says nothing about this chip (granule-edge nodata may sit far from it), so the
    NIR pixels are checked. Old-baseline products carry edge nodata on most scenes.
    """
    scene_nodata = get_nodata_from_metadata(item)
    if scene_nodata is not None and scene_nodata <= 0:
        return False, 0.0
    nir = item.assets.get("nir")
    if nodata_max >= 100 or nir is None:
        return False, None
    try:
        nodata_pct = calculate_nodata_percentage(nir.href, bbox)
    except Exception as e:
        if scene_nodata is not None and scene_nodata > nodata_max:
            log(
                f"  Skipping {short_date(item)}: nodata check failed ({e}), "
                f"scene reports {scene_nodata:.1f}% nodata"
            )
            return True, None
        log(f"  {item.id}: nodata check failed ({e}), continuing")
        return False, None
    if nodata_pct > nodata_max:
        log(f"  Skipping {short_date(item)}: {nodata_pct:.1f}% nodata in chip window")
        return True, nodata_pct
    return False, nodata_pct


@dataclass
class Sentinel2Source:
    """Sentinel-2 L2A scenes, rated per chip with the SCL band."""

    collection: str = "c1"
    backend: str = "parquet"

    name: ClassVar[str] = "sentinel-2"
    suffix: ClassVar[str] = "s2"
    title: ClassVar[str] = "Sentinel-2"
    default_workers: ClassVar[int] = 16
    reflectance_bands: ClassVar[frozenset[str]] = REFLECTANCE_BANDS

    def __post_init__(self) -> None:
        if self.backend not in SEARCH_BACKENDS:
            raise ValueError(
                f"Unknown search_backend {self.backend!r}; use 'parquet' or 'earth-search'."
            )

    @property
    def stac_host(self) -> str:
        return "parquet-mirror" if self.backend == "parquet" else "earthsearch"

    def search(
        self, bbox: tuple[float, float, float, float], center_date: datetime, buffer_days: int
    ) -> SearchResult:
        query = _query_parquet if self.backend == "parquet" else _query_stac
        result = query(
            bbox=bbox,
            center_date=center_date,
            cloud_cover_max=DEFAULT_CLOUD_COVER_SCENE,
            buffer_days=buffer_days,
            s2_collection=self.collection,
        )
        return SearchResult(
            items=result.items,
            description=[
                f"STAC Query: {result.catalog_url}",
                f"  Collection: {result.collection}",
                f"  Bbox: {result.bbox}",
                f"  Date range: {result.date_range}",
                f"  Cloud cover max: {result.cloud_cover_max}%",
            ],
        )

    def scene_cloud_cover(self, item: pystac.Item) -> float:
        return EOExtension.ext(item).cloud_cover or 0.0

    def assess(
        self,
        item: pystac.Item,
        bbox: tuple[float, float, float, float],
        nodata_max: float,
        log: Callable[[str], None],
    ) -> ChipAssessment | None:
        rejected, nodata = _chip_nodata(item, bbox, nodata_max, log)
        if rejected:
            return None

        scene_cc = self.scene_cloud_cover(item)
        if scene_cc < PIXEL_CHECK_SKIP_THRESHOLD:
            log(
                f"  {item.id}: scene cloud {scene_cc:.1f}% "
                f"(trusted, < {PIXEL_CHECK_SKIP_THRESHOLD}%)"
            )
            return ChipAssessment(scene_cc, "scene", nodata=nodata)

        scl = item.assets.get("scl")
        if scl is None:
            log(f"  {item.id}: no SCL asset, using scene cloud cover {scene_cc:.1f}%")
            return ChipAssessment(scene_cc, "scene", nodata=nodata)
        try:
            chip_cc = calculate_pixel_cloud_cover(cloud_href=scl.href, bbox=bbox, cloud_type="scl")
        except Exception as e:
            log(f"  {item.id}: chip check failed ({e}), using scene cloud cover")
            return ChipAssessment(scene_cc, "scene", nodata=nodata)
        log(f"  {item.id}: scene {scene_cc:.1f}% -> chip {chip_cc:.1f}%")
        return ChipAssessment(chip_cc, "pixel", cloud_mask="scl", nodata=nodata)

    def child_assets(self, item: pystac.Item) -> dict[str, pystac.Asset]:
        assets = {
            band: item.assets[band].clone() for band in CHILD_ITEM_BANDS if band in item.assets
        }
        if "cloud" in item.assets:
            assets["cloud_probability"] = item.assets["cloud"].clone()
        return assets

    def child_properties(self, item: pystac.Item) -> dict:
        properties: dict = {"constellation": "sentinel-2", "gsd": 10}
        platform = item.properties.get("platform")
        match = re.match(r"S2([A-D])_", item.id)
        if platform is None and match:
            platform = f"sentinel-2{match.group(1).lower()}"
        if platform:
            properties["platform"] = platform
        return properties

    def fetch(
        self,
        child: pystac.Item,
        child_path: Path | None,  # noqa: ARG002
        bands: list[str],
        log: Callable[[str], None],  # noqa: ARG002
    ) -> FetchResult:
        hrefs = {band: (child.assets[band].href, 1) for band in bands if band in child.assets}
        if not hrefs:
            return FetchResult("failed", error=_missing_bands_error(child, bands))
        return FetchResult("ready", hrefs)
