"""PlanetScope (PSScene) from the Planet Data and Orders APIs."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pystac
from pystac.extensions.eo import EOExtension
from shapely.geometry import box, shape

from ftw_dataset_tools.api.imagery.settings import DEFAULT_CLOUD_COVER_SCENE
from ftw_dataset_tools.api.imagery.sources.base import (
    ChipAssessment,
    FetchResult,
    ImagerySourceError,
    SearchResult,
    SourceUnavailableError,
    search_window,
    short_date,
)
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["DEFAULT_BUNDLE", "PLANET_BUNDLES", "PlanetScopeSource"]

PLANET_DATA_URL = "https://api.planet.com/data/v1"
ITEM_TYPE = "PSScene"

# Order bundle -> (analytic asset type, band names in file order)
PLANET_BUNDLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "analytic_sr_udm2": ("ortho_analytic_4b_sr", ("blue", "green", "red", "nir")),
    "analytic_8b_sr_udm2": (
        "ortho_analytic_8b_sr",
        ("coastal", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"),
    ),
}
DEFAULT_BUNDLE = "analytic_sr_udm2"

_DONE_STATES = frozenset({"success", "partial"})
_FAILED_STATES = frozenset({"failed", "cancelled"})
# Padding around the chip bbox in the clip AOI, so resampling has edge pixels.
_CLIP_PAD_DEGREES = 0.0003

_CLIENTS = threading.local()


def _bbox_polygon(bbox: tuple[float, ...] | list[float], pad: float = 0.0) -> dict:
    minx, miny, maxx, maxy = bbox
    return box(minx - pad, miny - pad, maxx + pad, maxy + pad).__geo_interface__


def _api_key() -> str:
    key = os.environ.get("PL_API_KEY")
    if not key:
        raise SourceUnavailableError("Set PL_API_KEY to use the planetscope source.")
    return key


def _planet_client() -> Any:
    """This thread's Planet SDK client."""
    client = getattr(_CLIENTS, "client", None)
    if client is not None:
        return client
    _api_key()
    try:
        from planet import Planet
    except ImportError as e:
        raise SourceUnavailableError(
            "The planetscope source needs the Planet SDK: pip install 'ftw-dataset-tools[planet]'"
        ) from e
    _CLIENTS.client = Planet()
    return _CLIENTS.client


def _to_item(feature: dict) -> pystac.Item:
    """A Data API search result as a pystac Item (cloud cover converted to %)."""
    props = feature["properties"]
    properties: dict = {}
    if props.get("cloud_cover") is not None:
        properties["eo:cloud_cover"] = round(props["cloud_cover"] * 100, 2)
    for key in ("clear_percent", "instrument", "satellite_id", "gsd"):
        if key in props:
            properties[f"pl:{key}"] = props[key]
    geometry = feature["geometry"]
    item = pystac.Item(
        id=feature["id"],
        geometry=geometry,
        bbox=list(shape(geometry).bounds),
        datetime=datetime.fromisoformat(props["acquired"].replace("Z", "+00:00")),
        properties=properties,
    )
    EOExtension.ext(item, add_if_missing=True)
    item.set_self_href(f"{PLANET_DATA_URL}/item-types/{ITEM_TYPE}/items/{feature['id']}")
    return item


def _outside_footprint_pct(item: pystac.Item, bbox: tuple[float, float, float, float]) -> float:
    chip = box(*bbox)
    return 100.0 * (1.0 - chip.intersection(shape(item.geometry)).area / chip.area)


def _scene_id(child: pystac.Item) -> str:
    scene_id = child.properties.get("ftw:scene_id")
    if scene_id:
        return scene_id
    via = child.get_single_link("via")
    if via is None:
        raise ImagerySourceError(f"{child.id} records no source scene")
    return via.get_href(transform_href=False).rstrip("/").rsplit("/", 1)[-1]


@dataclass
class PlanetScopeSource:
    """PlanetScope scenes, rated per chip with UDM2 and delivered as clipped orders."""

    bundle: str = DEFAULT_BUNDLE
    harmonize: bool = True
    wait: bool = True
    timeout_minutes: float = 60.0
    poll_seconds: float = 30.0

    name: ClassVar[str] = "planetscope"
    suffix: ClassVar[str] = "planet"
    title: ClassVar[str] = "PlanetScope"
    default_workers: ClassVar[int] = 4
    reflectance_bands: ClassVar[frozenset[str]] = frozenset(
        band for _, bands in PLANET_BUNDLES.values() for band in bands
    )

    def __post_init__(self) -> None:
        if self.bundle not in PLANET_BUNDLES:
            raise ValueError(
                f"Unknown Planet bundle {self.bundle!r}; use one of {list(PLANET_BUNDLES)}."
            )

    @property
    def stac_host(self) -> str:
        return "planet-data-api"

    def search(
        self, bbox: tuple[float, float, float, float], center_date: datetime, buffer_days: int
    ) -> SearchResult:
        start, end = search_window(center_date, buffer_days)
        asset_type = PLANET_BUNDLES[self.bundle][0]
        search_filter = {
            "type": "AndFilter",
            "config": [
                {"type": "GeometryFilter", "field_name": "geometry", "config": _bbox_polygon(bbox)},
                {
                    "type": "DateRangeFilter",
                    "field_name": "acquired",
                    "config": {
                        "gte": f"{start:%Y-%m-%dT%H:%M:%SZ}",
                        "lte": f"{end:%Y-%m-%dT%H:%M:%SZ}",
                    },
                },
                {
                    "type": "RangeFilter",
                    "field_name": "cloud_cover",
                    "config": {"lte": DEFAULT_CLOUD_COVER_SCENE / 100},
                },
                {
                    "type": "StringInFilter",
                    "field_name": "quality_category",
                    "config": ["standard"],
                },
                {
                    "type": "StringInFilter",
                    "field_name": "publishing_stage",
                    "config": ["finalized"],
                },
                {"type": "AssetFilter", "config": [asset_type, "ortho_udm2"]},
                {"type": "PermissionFilter", "config": ["assets:download"]},
            ],
        }
        features = _planet_client().data.search([ITEM_TYPE], search_filter=search_filter, limit=250)
        items = sorted((_to_item(feature) for feature in features), key=self.scene_cloud_cover)
        return SearchResult(
            items=items,
            description=[
                f"Planet Data API: {ITEM_TYPE} with {asset_type}",
                f"  Bbox: {bbox}",
                f"  Date range: {start.date()}/{end.date()}",
                f"  Cloud cover max: {DEFAULT_CLOUD_COVER_SCENE}%",
            ],
        )

    def scene_cloud_cover(self, item: pystac.Item) -> float:
        cloud = item.properties.get("eo:cloud_cover")
        return float(cloud) if cloud is not None else 0.0

    def _chip_clear_percent(self, item_id: str, bbox: tuple[float, float, float, float]) -> float:
        """Clear % of the chip from the scene's UDM2, via the Data API coverage endpoint."""
        import httpx

        url = f"{PLANET_DATA_URL}/item-types/{ITEM_TYPE}/items/{item_id}/coverage"
        for _ in range(5):
            response = httpx.post(
                url,
                params={"mode": "UDM2", "band": "clear"},
                json={"geometry": _bbox_polygon(bbox)},
                auth=(_api_key(), ""),
                timeout=60,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("status") == "complete":
                return float(body["clear_percent"])
            time.sleep(self.poll_seconds / 5)
        raise ImagerySourceError(f"UDM2 coverage for {item_id} did not complete")

    def assess(
        self,
        item: pystac.Item,
        bbox: tuple[float, float, float, float],
        nodata_max: float,
        log: Callable[[str], None],
    ) -> ChipAssessment | None:
        nodata = _outside_footprint_pct(item, bbox)
        if nodata > nodata_max:
            log(f"  Skipping {short_date(item)}: {nodata:.1f}% of chip outside the scene")
            return None
        scene_cc = self.scene_cloud_cover(item)
        try:
            chip_cc = 100.0 - self._chip_clear_percent(item.id, bbox)
        except Exception as e:
            log(f"  {item.id}: UDM2 check failed ({e}), using scene cloud cover")
            return ChipAssessment(scene_cc, "scene", nodata=nodata)
        log(f"  {item.id}: scene {scene_cc:.1f}% -> chip {chip_cc:.1f}% not clear (UDM2)")
        return ChipAssessment(chip_cc, "pixel", cloud_mask="udm2", nodata=nodata)

    def child_assets(self, item: pystac.Item) -> dict[str, pystac.Asset]:  # noqa: ARG002
        return {}

    def child_properties(self, item: pystac.Item) -> dict:
        props = item.properties
        properties: dict = {
            "constellation": "planetscope",
            "gsd": props.get("pl:gsd", 3.0),
            "ftw:planet_item_type": ITEM_TYPE,
            "ftw:planet_bundle": self.bundle,
        }
        if props.get("pl:satellite_id"):
            properties["platform"] = props["pl:satellite_id"]
        if props.get("pl:instrument"):
            properties["instruments"] = [props["pl:instrument"]]
        return properties

    def _submit(self, child: pystac.Item, bundle: str) -> str:
        tools: list[dict] = [{"clip": {"aoi": _bbox_polygon(child.bbox, _CLIP_PAD_DEGREES)}}]
        if self.harmonize:
            tools.append({"harmonize": {"target_sensor": "Sentinel-2"}})
        request = {
            "name": child.id,
            "products": [
                {"item_ids": [_scene_id(child)], "item_type": ITEM_TYPE, "product_bundle": bundle}
            ],
            "tools": tools,
        }
        return _planet_client().orders.create_order(request)["id"]

    def _order_state(self, order_id: str, log: Callable[[str], None]) -> str:
        deadline = time.monotonic() + (self.timeout_minutes * 60 if self.wait else 0)
        while True:
            state = _planet_client().orders.get_order(order_id)["state"]
            if state in _DONE_STATES | _FAILED_STATES or time.monotonic() >= deadline:
                return state
            log(f"Planet order {order_id}: {state}")
            time.sleep(self.poll_seconds)

    def _download(self, order_id: str, directory: Path) -> Path:
        paths = _planet_client().orders.download_order(
            order_id, directory=directory, overwrite=True, progress_bar=False
        )
        images = [Path(p) for p in paths if Path(p).suffix == ".tif" and "udm" not in Path(p).name]
        if not images:
            raise ImagerySourceError(f"Planet order {order_id} delivered no analytic image")
        return images[0]

    def fetch(
        self,
        child: pystac.Item,
        child_path: Path | None,
        bands: list[str],
        log: Callable[[str], None],
    ) -> FetchResult:
        props = child.properties
        bundle = props.get("ftw:planet_bundle", self.bundle)
        band_names = PLANET_BUNDLES[bundle][1]
        missing = [band for band in bands if band not in band_names]
        if missing:
            return FetchResult("failed", error=f"{bundle} has no bands {missing}")
        if child_path is None:
            return FetchResult("failed", error="PlanetScope downloads need the child item path")

        try:
            order_id = props.get("ftw:planet_order_id")
            if order_id is None:
                order_id = self._submit(child, bundle)
                props["ftw:planet_order_id"] = order_id
                props["ftw:harmonized"] = self.harmonize
                write_item(child, child_path)
                log(f"Submitted Planet order {order_id}")

            state = self._order_state(order_id, log)
            if state in _FAILED_STATES:
                props.pop("ftw:planet_order_id")
                write_item(child, child_path)
                return FetchResult("failed", error=f"Planet order {order_id} {state}")
            if state not in _DONE_STATES:
                return FetchResult("pending", error=f"Planet order {order_id} is {state}")

            workdir = child_path.parent / ".planet" / order_id
            image = self._download(order_id, workdir)
        except SourceUnavailableError:
            raise
        except Exception as e:
            return FetchResult("failed", error=f"Planet order failed: {e}")

        return FetchResult(
            "ready",
            {band: (str(image), band_names.index(band) + 1) for band in bands},
            cleanup=[workdir],
        )
