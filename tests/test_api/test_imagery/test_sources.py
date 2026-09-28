"""Tests for imagery sources, per-source catalog layout and scene provenance."""

from __future__ import annotations

import json
import sys
import urllib.error
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pystac
import pytest
import rasterio
from click.testing import CliRunner
from pystac.extensions.eo import EOExtension
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.imagery import (
    catalog_ops,
    download_workflow,
    image_download,
    naming,
    scene_selection,
)
from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates
from ftw_dataset_tools.api.imagery.image_download import DownloadResult, download_and_clip_scene
from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult, SelectedScene
from ftw_dataset_tools.api.imagery.sources import (
    PlanetScopeSource,
    Sentinel2Source,
    SourceUnavailableError,
    build_source,
    planetscope,
    sentinel2,
)
from ftw_dataset_tools.api.imagery.sources.base import (
    ChipAssessment,
    FetchResult,
    ImagerySourceError,
    SearchResult,
    short_date,
)
from ftw_dataset_tools.api.imagery.stac_child_items import (
    attach_existing_seasons,
    create_child_items_from_selection,
)
from ftw_dataset_tools.api.stac_items import write_item
from ftw_dataset_tools.cli import cli

from .conftest import create_mock_s2_assets, create_mock_stac_item

BBOX = (10.0, 50.0, 10.01, 50.01)


def _planet_feature(item_id: str, acquired: str, cloud: float, bbox=BBOX) -> dict:
    minx, miny, maxx, maxy = bbox
    return {
        "id": item_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
        },
        "properties": {
            "acquired": acquired,
            "cloud_cover": cloud,
            "clear_percent": 100 - cloud * 100,
            "instrument": "PSB.SD",
            "satellite_id": "24a1",
            "gsd": 3.7,
        },
    }


def _s2_item(item_id: str, cloud: float, **properties: object) -> pystac.Item:
    item = create_mock_stac_item(
        item_id,
        bbox=BBOX,
        properties={"eo:cloud_cover": cloud, **properties},
        assets=create_mock_s2_assets(),
    )
    EOExtension.ext(item, add_if_missing=True)
    return item


def _write_collection(dataset_dir: Path) -> None:
    pystac.Collection(
        id="test-dataset",
        description="Test dataset",
        extent=pystac.Extent(
            pystac.SpatialExtent([list(BBOX)]),
            pystac.TemporalExtent([[datetime(2023, 1, 1, tzinfo=UTC), None]]),
        ),
    ).save_object(include_self_link=False, dest_href=str(dataset_dir / "collection.json"))


class TestRegistryAndNaming:
    def test_build_source_passes_only_relevant_options(self) -> None:
        s2 = build_source("sentinel-2", s2_collection="old-baseline", search_backend="earth-search")
        planet = build_source("planetscope", planet_bundle="analytic_8b_sr_udm2", planet_wait=False)

        assert (s2.collection, s2.backend) == ("old-baseline", "earth-search")
        assert (planet.bundle, planet.wait) == ("analytic_8b_sr_udm2", False)

    def test_unknown_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown imagery source"):
            build_source("landsat")

    def test_unknown_planet_bundle_rejected(self) -> None:
        with pytest.raises(ValueError, match="Planet bundle"):
            PlanetScopeSource(bundle="visual")

    @pytest.mark.parametrize("source", ["sentinel-2", "planetscope"])
    def test_child_id_and_image_name_round_trip(self, source: str) -> None:
        chip = "ftw-34UFF1628_2024"
        ref = naming.parse_child_id(naming.child_item_id(chip, "harvest", source))
        image = naming.parse_image_stem(naming.image_filename(chip, "harvest", source)[:-4])

        assert ref == image == naming.ChildRef(chip, "harvest", source)

    def test_chip_item_is_not_a_child(self) -> None:
        assert naming.parse_child_id("ftw-34UFF1628_2024") is None
        assert naming.parse_child_id("chip_planting_landsat") is None

    def test_default_source_keeps_legacy_names(self) -> None:
        assert naming.child_item_id("c", "planting") == "c_planting_s2"
        assert naming.parent_asset_key("planting", "image") == "planting_image"
        assert naming.parent_asset_key("planting", "image", "planetscope") == (
            "planting_image_planet"
        )


class TestSentinel2Source:
    def test_child_properties_derive_platform_from_scene_id(self) -> None:
        item = create_mock_stac_item("S2B_32TNT_20230514_0_L2A")
        assert Sentinel2Source().child_properties(item) == {
            "constellation": "sentinel-2",
            "gsd": 10,
            "platform": "sentinel-2b",
        }

    def test_fetch_returns_remote_band_hrefs(self) -> None:
        child = create_mock_stac_item("c_planting_s2", assets=create_mock_s2_assets())
        fetched = Sentinel2Source().fetch(child, None, ["red", "nir"], lambda _m: None)

        assert fetched.status == "ready"
        assert fetched.bands == {
            "red": ("https://example.com/data/red.tif", 1),
            "nir": ("https://example.com/data/nir.tif", 1),
        }

    @pytest.mark.parametrize(
        ("error", "retryable"),
        [
            (urllib.error.HTTPError("https://x", 503, "busy", {}, None), True),
            (urllib.error.HTTPError("https://x", 404, "missing", {}, None), False),
            (urllib.error.URLError("down"), True),
            (ValueError("bad"), False),
        ],
    )
    def test_only_transient_errors_are_retried(self, error: Exception, retryable: bool) -> None:
        assert sentinel2._is_retryable(error) is retryable

    def test_earth_search_retries_then_sorts_by_cloud(self, monkeypatch) -> None:
        calls = []

        def search(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise urllib.error.HTTPError("https://x", 503, "busy", {}, None)
            return SimpleNamespace(items=lambda: iter([_s2_item("b", 30.0), _s2_item("a", 2.0)]))

        monkeypatch.setattr(
            sentinel2, "_get_stac_client", lambda _url: SimpleNamespace(search=search)
        )
        monkeypatch.setattr(sentinel2.time, "sleep", lambda _s: None)

        result = sentinel2._query_stac(BBOX, datetime(2023, 5, 14, tzinfo=UTC), 20, 7)

        assert [item.id for item in result.items] == ["a", "b"]
        assert len(calls) == 2
        assert result.collection == "sentinel-2-c1-l2a"

    def test_earth_search_raises_permanent_errors_at_once(self, monkeypatch) -> None:
        def search(**_kwargs):
            raise urllib.error.HTTPError("https://x", 404, "missing", {}, None)

        monkeypatch.setattr(
            sentinel2, "_get_stac_client", lambda _url: SimpleNamespace(search=search)
        )

        with pytest.raises(urllib.error.HTTPError):
            sentinel2._query_stac(BBOX, datetime(2023, 5, 14, tzinfo=UTC), 20, 7)

    def test_nodata_is_not_checked_without_a_nir_band(self) -> None:
        item = _s2_item("S2B_x", 1.0, **{"s2:nodata_pixel_percentage": 5.0})
        del item.assets["nir"]

        assert sentinel2._chip_nodata(item, BBOX, 10.0, lambda _m: None) == (False, None)

    def test_cloudy_scene_is_rated_with_scl(self, monkeypatch) -> None:
        item = _s2_item("S2B_x", 40.0, **{"s2:nodata_pixel_percentage": 0.0})
        monkeypatch.setattr(sentinel2, "calculate_pixel_cloud_cover", lambda **_kw: 3.0)

        assessment = Sentinel2Source().assess(item, BBOX, 5.0, lambda _m: None)

        assert assessment == ChipAssessment(3.0, "pixel", cloud_mask="scl", nodata=0.0)

    def test_scl_failure_falls_back_to_scene_cloud(self, monkeypatch) -> None:
        item = _s2_item("S2B_x", 40.0, **{"s2:nodata_pixel_percentage": 0.0})

        def fail(**_kwargs):
            raise OSError("unreadable")

        monkeypatch.setattr(sentinel2, "calculate_pixel_cloud_cover", fail)

        assessment = Sentinel2Source().assess(item, BBOX, 5.0, lambda _m: None)

        assert (assessment.cloud_cover, assessment.cloud_cover_source) == (40.0, "scene")

    def test_scene_without_scl_uses_scene_cloud(self) -> None:
        item = _s2_item("S2B_x", 40.0, **{"s2:nodata_pixel_percentage": 0.0})
        del item.assets["scl"]

        assessment = Sentinel2Source().assess(item, BBOX, 5.0, lambda _m: None)

        assert (assessment.cloud_cover, assessment.cloud_cover_source) == (40.0, "scene")


@pytest.fixture
def fake_planet(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A stand-in for the Planet SDK client."""
    state = SimpleNamespace(features=[], created=[], states=["success"], search_filter=None)

    def search(*_args, search_filter, **_kwargs):
        state.search_filter = search_filter
        return iter(state.features)

    def create_order(request):
        state.created.append(request)
        return {"id": f"order-{len(state.created)}"}

    def get_order(_order_id):
        return {"state": state.states.pop(0) if len(state.states) > 1 else state.states[0]}

    def download_order(_order_id, directory, **_kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        image = directory / "scene_3B_AnalyticMS_SR_harmonized_clip.tif"
        udm = directory / "scene_3B_udm2_clip.tif"
        image.write_bytes(b"")
        udm.write_bytes(b"")
        return [udm, image]

    client = SimpleNamespace(
        data=SimpleNamespace(search=search),
        orders=SimpleNamespace(
            create_order=create_order, get_order=get_order, download_order=download_order
        ),
    )
    monkeypatch.setattr(planetscope, "_planet_client", lambda: client)
    return state


def _planet_child(tmp_path: Path, **properties: object) -> tuple[pystac.Item, Path]:
    child = create_mock_stac_item(
        "chip_001_planting_planet",
        bbox=BBOX,
        properties={
            "ftw:source": "planetscope",
            "ftw:scene_id": "20230514_101010_24a1",
            "ftw:planet_bundle": "analytic_sr_udm2",
            **properties,
        },
    )
    path = tmp_path / "chip_001_planting_planet.json"
    write_item(child, path)
    return child, path


class TestPlanetScopeSource:
    def test_search_converts_cloud_to_percent_and_sorts(self, fake_planet) -> None:
        fake_planet.features = [
            _planet_feature("b", "2023-05-15T10:00:00Z", 0.30),
            _planet_feature("a", "2023-05-14T10:00:00Z", 0.05),
        ]
        result = PlanetScopeSource().search(BBOX, datetime(2023, 5, 14, tzinfo=UTC), 7)

        assert [item.id for item in result.items] == ["a", "b"]
        assert result.items[0].properties["eo:cloud_cover"] == 5.0
        filters = {f["type"] for f in fake_planet.search_filter["config"]}
        assert {"AssetFilter", "PermissionFilter", "DateRangeFilter"} <= filters

    def test_assess_rejects_chip_partly_outside_scene(self) -> None:
        half = (10.005, 50.0, 10.02, 50.01)
        item = planetscope._to_item(_planet_feature("a", "2023-05-14T10:00:00Z", 0.0, half))

        assert PlanetScopeSource().assess(item, BBOX, 0.0, lambda _m: None) is None

    def test_assess_uses_udm2_clear_percent(self, monkeypatch) -> None:
        item = planetscope._to_item(_planet_feature("a", "2023-05-14T10:00:00Z", 0.4))
        monkeypatch.setattr(PlanetScopeSource, "_chip_clear_percent", lambda *_a: 97.5)

        assessment = PlanetScopeSource().assess(item, BBOX, 0.0, lambda _m: None)

        assert assessment == ChipAssessment(2.5, "pixel", cloud_mask="udm2", nodata=0.0)

    def test_assess_falls_back_to_scene_cloud(self, monkeypatch) -> None:
        item = planetscope._to_item(_planet_feature("a", "2023-05-14T10:00:00Z", 0.4))

        def fail(*_args):
            raise OSError("no udm2")

        monkeypatch.setattr(PlanetScopeSource, "_chip_clear_percent", fail)
        assessment = PlanetScopeSource().assess(item, BBOX, 0.0, lambda _m: None)

        assert (assessment.cloud_cover, assessment.cloud_cover_source) == (40.0, "scene")

    def test_fetch_orders_once_and_maps_bands(self, fake_planet, tmp_path: Path) -> None:
        child, path = _planet_child(tmp_path)
        fetched = PlanetScopeSource(poll_seconds=0).fetch(
            child, path, ["red", "nir"], lambda _m: None
        )

        assert fetched.status == "ready"
        assert Path(fetched.bands["red"][0]).name.endswith("AnalyticMS_SR_harmonized_clip.tif")
        assert (fetched.bands["red"][1], fetched.bands["nir"][1]) == (3, 4)
        request = fake_planet.created[0]
        assert request["products"][0]["item_ids"] == ["20230514_101010_24a1"]
        assert [next(iter(tool)) for tool in request["tools"]] == ["clip", "harmonize"]
        saved = json.loads(path.read_text())["properties"]
        assert saved["ftw:planet_order_id"] == "order-1"

    def test_running_order_is_pending_and_resumed_without_reordering(
        self, fake_planet, tmp_path: Path
    ) -> None:
        fake_planet.states = ["running"]
        child, path = _planet_child(tmp_path)
        source = PlanetScopeSource(wait=False)

        first = source.fetch(child, path, ["red"], lambda _m: None)
        fake_planet.states = ["success"]
        resumed = pystac.Item.from_file(str(path))
        second = source.fetch(resumed, path, ["red"], lambda _m: None)

        assert first.status == "pending"
        assert second.status == "ready"
        assert len(fake_planet.created) == 1

    def test_failed_order_is_forgotten_so_the_next_run_reorders(
        self, fake_planet, tmp_path: Path
    ) -> None:
        fake_planet.states = ["failed"]
        child, path = _planet_child(tmp_path)

        fetched = PlanetScopeSource(wait=False).fetch(child, path, ["red"], lambda _m: None)

        assert fetched.status == "failed"
        assert "ftw:planet_order_id" not in json.loads(path.read_text())["properties"]

    def test_band_missing_from_bundle_fails(self, tmp_path: Path) -> None:
        child, path = _planet_child(tmp_path)
        fetched = PlanetScopeSource().fetch(child, path, ["swir16"], lambda _m: None)

        assert fetched.status == "failed"
        assert "swir16" in fetched.error

    def test_missing_api_key_is_reported(self, monkeypatch) -> None:
        monkeypatch.delenv("PL_API_KEY", raising=False)
        monkeypatch.setattr(planetscope, "_CLIENTS", SimpleNamespace())

        with pytest.raises(SourceUnavailableError, match="PL_API_KEY"):
            planetscope._planet_client()

    def test_client_is_created_once_per_thread(self, monkeypatch) -> None:
        sdk = ModuleType("planet")
        sdk.Planet = lambda: object()
        monkeypatch.setenv("PL_API_KEY", "key")
        monkeypatch.setattr(planetscope, "_CLIENTS", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "planet", sdk)

        assert planetscope._planet_client() is planetscope._planet_client()

    def test_missing_sdk_is_reported(self, monkeypatch) -> None:
        monkeypatch.setenv("PL_API_KEY", "key")
        monkeypatch.setattr(planetscope, "_CLIENTS", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "planet", None)

        with pytest.raises(SourceUnavailableError, match="Planet SDK"):
            planetscope._planet_client()

    def test_sparse_search_result_and_child_properties(self) -> None:
        full = planetscope._to_item(_planet_feature("a", "2023-05-14T10:00:00Z", 0.1))
        feature = _planet_feature("b", "2023-05-14T10:00:00Z", 0.1)
        feature["properties"] = {"acquired": "2023-05-14T10:00:00Z"}
        sparse = planetscope._to_item(feature)
        source = PlanetScopeSource()

        assert source.child_properties(full)["platform"] == "24a1"
        assert source.child_properties(full)["instruments"] == ["PSB.SD"]
        assert "eo:cloud_cover" not in sparse.properties
        assert source.scene_cloud_cover(sparse) == 0.0
        assert source.child_properties(sparse) == {
            "constellation": "planetscope",
            "gsd": 3.0,
            "ftw:planet_item_type": "PSScene",
            "ftw:planet_bundle": "analytic_sr_udm2",
        }

    def test_scene_id_falls_back_to_the_via_link(self) -> None:
        child = create_mock_stac_item("chip_001_planting_planet", bbox=BBOX)
        child.add_link(pystac.Link("via", f"{planetscope.PLANET_DATA_URL}/items/scene-9"))

        assert planetscope._scene_id(child) == "scene-9"

        child.clear_links("via")
        with pytest.raises(ImagerySourceError, match="records no source scene"):
            planetscope._scene_id(child)

    def test_udm2_coverage_is_polled_until_complete(self, monkeypatch) -> None:
        bodies = [{"status": "running"}, {"status": "complete", "clear_percent": 88.5}]

        def post(*_args, **_kwargs):
            body = bodies.pop(0)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

        monkeypatch.setenv("PL_API_KEY", "key")
        monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post))

        assert PlanetScopeSource(poll_seconds=0)._chip_clear_percent("a", BBOX) == 88.5

    def test_udm2_coverage_that_never_completes_raises(self, monkeypatch) -> None:
        response = SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"status": "running"}
        )
        monkeypatch.setenv("PL_API_KEY", "key")
        monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=lambda *_a, **_k: response))

        with pytest.raises(ImagerySourceError, match="did not complete"):
            PlanetScopeSource(poll_seconds=0)._chip_clear_percent("a", BBOX)

    def test_waits_for_a_running_order(self, fake_planet, tmp_path: Path) -> None:
        fake_planet.states = ["queued", "running", "success"]
        child, path = _planet_child(tmp_path)
        messages: list[str] = []

        fetched = PlanetScopeSource(poll_seconds=0, harmonize=False).fetch(
            child, path, ["red"], messages.append
        )

        assert fetched.status == "ready"
        assert "Planet order order-1: running" in messages
        assert [next(iter(tool)) for tool in fake_planet.created[0]["tools"]] == ["clip"]

    @pytest.mark.usefixtures("fake_planet")
    def test_order_without_analytic_image_fails(self, tmp_path: Path) -> None:
        client = planetscope._planet_client()
        client.orders.download_order = lambda _id, directory, **_kw: [directory / "x_udm2.tif"]
        child, path = _planet_child(tmp_path)

        fetched = PlanetScopeSource().fetch(child, path, ["red"], lambda _m: None)

        assert fetched.status == "failed"
        assert "delivered no analytic image" in fetched.error

    def test_fetch_needs_the_child_path(self, tmp_path: Path) -> None:
        child, _path = _planet_child(tmp_path)

        fetched = PlanetScopeSource().fetch(child, None, ["red"], lambda _m: None)

        assert (fetched.status, fetched.error) == (
            "failed",
            "PlanetScope downloads need the child item path",
        )

    def test_missing_credentials_stop_the_fetch(self, monkeypatch, tmp_path: Path) -> None:
        def unavailable():
            raise SourceUnavailableError("Set PL_API_KEY")

        monkeypatch.setattr(planetscope, "_planet_client", unavailable)
        child, path = _planet_child(tmp_path)

        with pytest.raises(SourceUnavailableError):
            PlanetScopeSource().fetch(child, path, ["red"], lambda _m: None)


@dataclass
class FakeSource:
    """A source whose candidates and chip ratings are given up front."""

    items: list[pystac.Item]
    cloud: dict[str, float]
    band_file: Path | None = None
    searches: list[int] = field(default_factory=list)
    cleanup: list[Path] = field(default_factory=list)

    name: ClassVar[str] = "planetscope"
    suffix: ClassVar[str] = "planet"
    title: ClassVar[str] = "PlanetScope"
    default_workers: ClassVar[int] = 1
    reflectance_bands: ClassVar[frozenset[str]] = frozenset({"red", "nir"})
    stac_host: ClassVar[str] = "fake"

    def search(self, _bbox, _center, buffer_days) -> SearchResult:
        self.searches.append(buffer_days)
        return SearchResult(self.items)

    def scene_cloud_cover(self, item) -> float:
        return item.properties.get("eo:cloud_cover", 0.0)

    def assess(self, item, _bbox, _nodata_max, _log) -> ChipAssessment:
        return ChipAssessment(self.cloud[item.id], "pixel", cloud_mask="udm2", nodata=0.0)

    def child_assets(self, _item) -> dict:
        return {}

    def child_properties(self, _item) -> dict:
        return {"constellation": "planetscope"}

    def fetch(self, _child, _path, bands, _log) -> FetchResult:
        return FetchResult(
            "ready",
            {band: (str(self.band_file), i + 1) for i, band in enumerate(bands)},
            cleanup=self.cleanup,
        )


def _scene_item(item_id: str, day: int, scene_cloud: float) -> pystac.Item:
    return create_mock_stac_item(
        item_id,
        bbox=BBOX,
        dt=datetime(2021, 6, day, 10, tzinfo=UTC),
        properties={"eo:cloud_cover": scene_cloud},
    )


class TestSelectionWithASource:
    @pytest.fixture(autouse=True)
    def _crop_calendar(self, monkeypatch):
        monkeypatch.setattr(
            scene_selection,
            "get_crop_calendar_dates",
            lambda _bbox, on_progress=None: CropCalendarDates(160, 270),  # noqa: ARG005
        )

    def test_selection_is_driven_by_the_source(self) -> None:
        source = FakeSource(
            items=[_scene_item("clear", 9, 1.0), _scene_item("cloudy", 8, 0.5)],
            cloud={"clear": 0.5, "cloudy": 50.0},
        )
        result = scene_selection.select_scenes_for_chip(
            "chip", BBOX, 2021, source=source, num_buffer_expansions=0
        )

        assert result.success
        assert result.planting_scene.id == "clear"
        assert result.planting_scene.cloud_mask == "udm2"
        assert result.selection_params["stac_host"] == "fake"

    def test_record_candidates_lists_every_clear_scene(self) -> None:
        source = FakeSource(
            items=[
                _scene_item("best", 12, 0.1),
                _scene_item("cloudy", 8, 0.2),
                _scene_item("early", 5, 0.3),
            ],
            cloud={"best": 0.0, "cloudy": 60.0, "early": 1.0},
        )
        result = scene_selection.select_scenes_for_chip(
            "chip", BBOX, 2021, source=source, record_candidates=True, num_buffer_expansions=0
        )

        scene = result.planting_scene
        assert scene.id == "best"
        assert [c["scene_id"] for c in scene.clear_candidates] == ["early", "best"]

    def test_missing_season_widens_its_window(self) -> None:
        source = FakeSource(items=[_scene_item("cloudy", 8, 0.2)], cloud={"cloudy": 60.0})
        result = scene_selection.select_scenes_for_chip(
            "chip",
            BBOX,
            2021,
            source=source,
            buffer_days=7,
            buffer_expansion_size=7,
            num_buffer_expansions=1,
        )

        assert not result.success
        assert result.expansions_performed == 1
        assert result.planting_buffer_used == 14

    def test_only_the_missing_season_is_searched_again(self) -> None:
        @dataclass
        class PlantingOnly(FakeSource):
            def search(self, bbox, center, buffer_days) -> SearchResult:
                result = super().search(bbox, center, buffer_days)
                return result if center.month < 8 else SearchResult([])

        source = PlantingOnly(items=[_scene_item("clear", 9, 0.1)], cloud={"clear": 0.0})
        result = scene_selection.select_scenes_for_chip(
            "chip",
            BBOX,
            2021,
            source=source,
            buffer_days=7,
            buffer_expansion_size=7,
            num_buffer_expansions=1,
        )

        assert result.planting_scene.id == "clear"
        assert result.skipped_reason == "No cloud-free harvest scene found"
        assert (result.planting_buffer_used, result.harvest_buffer_used) == (7, 14)
        assert source.searches == [7, 7, 14]

    def test_query_error_stops_the_search(self) -> None:
        source = FakeSource(items=[], cloud={})
        result = scene_selection.select_scenes_for_chip("chip", BBOX, 2999, source=source)

        assert result.skipped_reason.startswith("Planting query error: Query date range")
        assert source.searches == []


class TestSceneHelpers:
    def test_datetime_falls_back_to_start_datetime(self) -> None:
        item = _scene_item("a", 9, 0.0)
        item.datetime = None
        item.properties["start_datetime"] = "2021-06-09T10:00:00Z"

        assert scene_selection._get_item_datetime(item) == datetime(2021, 6, 9, 10, tzinfo=UTC)

    @pytest.mark.parametrize(
        ("start", "match"), [(None, "no datetime"), (20210609, "invalid start_datetime")]
    )
    def test_unusable_datetime_raises(self, start, match: str) -> None:
        item = _scene_item("a", 9, 0.0)
        item.datetime = None
        item.properties.pop("datetime", None)
        item.properties["start_datetime"] = start

        with pytest.raises(ValueError, match=match):
            scene_selection._get_item_datetime(item)

    def test_undated_candidate_is_skipped(self) -> None:
        item = _scene_item("a", 9, 0.0)
        item.datetime = None
        messages: list[str] = []

        scene = scene_selection._select_best_scene(
            [item], "planting", BBOX, source=FakeSource([], {"a": 0.0}), on_progress=messages.append
        )

        assert scene is None
        assert any("a: skipping" in message for message in messages)

    @pytest.mark.parametrize(
        ("planting", "harvest", "reason"),
        [
            (None, "scene", "No cloud-free planting scene found"),
            ("scene", None, "No cloud-free harvest scene found"),
            ("scene", "scene", None),
        ],
    )
    def test_skip_reason_names_the_missing_season(self, planting, harvest, reason) -> None:
        assert scene_selection._skip_reason(planting, harvest) == reason

    def test_short_date_of_an_undated_item_is_its_id(self) -> None:
        item = _scene_item("20230514_101010_24a1", 9, 0.0)
        item.datetime = None

        assert short_date(item) == "20230514_101010"


def _selection(source, scene_item: pystac.Item, cloud: float) -> SceneSelectionResult:
    def scene(season: str) -> SelectedScene:
        return SelectedScene(
            item=scene_item,
            season=season,
            cloud_cover=cloud,
            datetime=scene_item.datetime,
            stac_url=scene_item.get_self_href() or "",
            cloud_cover_source="pixel",
            cloud_mask="udm2" if source else "scl",
            nodata=0.0,
            scene_cloud_cover=12.345,
        )

    return SceneSelectionResult(
        chip_id="chip_001",
        bbox=BBOX,
        year=2023,
        crop_calendar=CropCalendarDates(150, 270),
        planting_scene=scene("planting"),
        harvest_scene=scene("harvest"),
        planting_buffer_used=14,
        harvest_buffer_used=28,
        source=source,
    )


@pytest.fixture
def two_source_chip(tmp_path: Path) -> tuple[Path, pystac.Item]:
    """A chip selected from Sentinel-2 and then PlanetScope."""
    chip_dir = tmp_path / "chips" / "32TNT" / "chip_001"
    chip_dir.mkdir(parents=True)
    parent = create_mock_stac_item("chip_001", bbox=BBOX)
    parent.set_self_href(str(chip_dir / "chip_001.json"))

    s2_item = create_mock_stac_item(
        "S2B_32TNT_20230514_0_L2A", bbox=BBOX, assets=create_mock_s2_assets()
    )
    planet_item = planetscope._to_item(
        _planet_feature("20230516_24a1", "2023-05-16T10:00:00Z", 0.1)
    )
    create_child_items_from_selection(chip_dir, parent, _selection(None, s2_item, 0.4), 2023, 2, 14)
    create_child_items_from_selection(
        chip_dir, parent, _selection(PlanetScopeSource(), planet_item, 1.2), 2023, 2, 14
    )
    return chip_dir, parent


class TestPerSourceCatalog:
    def test_child_records_scene_provenance(self, two_source_chip) -> None:
        chip_dir, _parent = two_source_chip
        props = json.loads((chip_dir / "chip_001_planting_planet.json").read_text())["properties"]

        assert props["ftw:source"] == "planetscope"
        assert props["ftw:scene_id"] == "20230516_24a1"
        assert props["eo:cloud_cover"] == 1.2
        assert props["ftw:scene_cloud_cover"] == 12.35
        assert props["ftw:cloud_cover_source"] == "pixel"
        assert props["ftw:cloud_mask"] == "udm2"
        assert props["ftw:buffer_used"] == 14
        assert props["ftw:planet_bundle"] == "analytic_sr_udm2"

    def test_sources_coexist_on_the_parent(self, two_source_chip) -> None:
        _chip_dir, parent = two_source_chip
        planting = [link for link in parent.links if link.rel == "ftw:planting"]

        assert {naming.link_source(link) for link in planting} == {"sentinel-2", "planetscope"}
        assert parent.properties["ftw:planting_cloud_cover"] == 0.4
        assert catalog_ops.has_existing_scenes(parent, "sentinel-2")
        assert catalog_ops.has_existing_scenes(parent, "planetscope")

    def test_clearing_one_source_keeps_the_other(self, two_source_chip) -> None:
        chip_dir, parent = two_source_chip
        (chip_dir / "chip_001_planting_image_planet.tif").write_bytes(b"")
        (chip_dir / "chip_001_planting_image_s2.tif").write_bytes(b"")

        result = catalog_ops.clear_chip_selections(parent, "planetscope")

        assert result.stac_items_deleted == 2
        assert (chip_dir / "chip_001_planting_s2.json").exists()
        assert (chip_dir / "chip_001_planting_image_s2.tif").exists()
        assert not (chip_dir / "chip_001_planting_image_planet.tif").exists()
        assert catalog_ops.has_existing_scenes(parent, "sentinel-2")
        assert not catalog_ops.has_existing_scenes(parent, "planetscope")

    def test_rebuilt_parent_gets_both_sources_back(self, two_source_chip) -> None:
        chip_dir, _parent = two_source_chip
        fresh = create_mock_stac_item("chip_001", bbox=BBOX)

        assert attach_existing_seasons(fresh, chip_dir) == ["planting", "harvest"]
        assert catalog_ops.has_existing_scenes(fresh, "sentinel-2")
        assert catalog_ops.has_existing_scenes(fresh, "planetscope")

    def test_rebuilt_parent_gets_the_downloaded_planet_image(self, two_source_chip) -> None:
        chip_dir, _parent = two_source_chip
        child_path = chip_dir / "chip_001_planting_planet.json"
        child = pystac.Item.from_file(str(child_path))
        _write_band_file(chip_dir / "chip_001_planting_image_planet.tif")
        child.add_asset("image", pystac.Asset(href="./chip_001_planting_image_planet.tif"))
        write_item(child, child_path)
        fresh = create_mock_stac_item("chip_001", bbox=BBOX)

        attach_existing_seasons(fresh, chip_dir)

        asset = fresh.assets["planting_image_planet"]
        assert asset.title == "Planting season PlanetScope imagery"

    def test_imagery_stats_per_source(self, two_source_chip) -> None:
        _chip_dir, parent = two_source_chip
        planet = catalog_ops.get_imagery_stats([parent], "planetscope")
        parent.properties.pop("ftw:planting_cloud_cover")
        parent.properties.pop("ftw:harvest_cloud_cover")
        s2 = catalog_ops.get_imagery_stats([parent])

        assert (planet.with_imagery, planet.planting_cloud_covers) == (1, [])
        assert (s2.with_imagery, s2.planting_cloud_covers, s2.harvest_cloud_covers) == (1, [], [])

    def test_clearing_the_default_source_keeps_the_shared_crop_calendar(
        self, two_source_chip
    ) -> None:
        _chip_dir, parent = two_source_chip

        catalog_ops.clear_chip_selections(parent, "sentinel-2")

        assert parent.properties["ftw:planting_day"] == 150
        assert "ftw:planting_cloud_cover" not in parent.properties
        assert catalog_ops.has_existing_scenes(parent, "planetscope")

    def test_child_records_clear_candidates(self, tmp_path: Path) -> None:
        parent = create_mock_stac_item("chip_002", bbox=BBOX)
        item = planetscope._to_item(_planet_feature("scene-1", "2023-05-16T10:00:00Z", 0.1))
        result = _selection(PlanetScopeSource(), item, 1.2)
        candidates = [{"scene_id": "scene-1", "datetime": "2023-05-16", "cloud_cover": 1.2}]
        result.planting_scene.clear_candidates = candidates

        create_child_items_from_selection(tmp_path, parent, result, 2023, 2, 14)

        child = json.loads((tmp_path / "chip_002_planting_planet.json").read_text())
        assert child["properties"]["ftw:clear_candidates"] == candidates

    def test_selection_without_scenes_sets_no_temporal_range(self, tmp_path: Path) -> None:
        parent = create_mock_stac_item("chip_002", bbox=BBOX)
        result = SceneSelectionResult("chip_002", BBOX, 2023, CropCalendarDates(150, 270))

        create_child_items_from_selection(tmp_path, parent, result, 2023, 2, 14)

        assert parent.properties["ftw:planting_day"] == 150
        assert "start_datetime" not in parent.properties


class TestPerSourceDownload:
    @pytest.mark.usefixtures("two_source_chip")
    def test_find_child_items_filters_by_source(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.download_workflow import (
            build_download_task,
            find_child_items,
        )

        planet = find_child_items(tmp_path, source="planetscope")
        task = build_download_task(*planet[0])

        assert len(find_child_items(tmp_path)) == 4
        assert {item.id for item, _ in planet} == {
            "chip_001_planting_planet",
            "chip_001_harvest_planet",
        }
        assert task.source == "planetscope"
        assert task.output_filename.endswith("_image_planet.tif")

    @pytest.mark.usefixtures("two_source_chip")
    def test_pending_order_counts_as_skipped(self, tmp_path, monkeypatch):
        from ftw_dataset_tools.api.imagery import download_workflow

        def pending(*, scene, output_path, **_kwargs) -> DownloadResult:
            return DownloadResult(
                output_path, scene.id, scene.season, [], 0, 0, "", False, "order running", True
            )

        monkeypatch.setattr(download_workflow, "download_and_clip_scene", pending)
        result = download_workflow.download_imagery_for_catalog(
            tmp_path, show_progress_bar=False, workers=1, source="planetscope"
        )

        assert (result.skipped, result.failed, result.successful) == (2, 0, 0)

    def test_clip_writes_provenance_tags_and_reads_band_indices(self, tmp_path: Path) -> None:
        output = tmp_path / "chip_001_planting_image_planet.tif"
        band_file = _write_band_file(tmp_path / "order.tif")
        result = _clip(output, FakeSource([], {}, band_file=band_file))

        assert result.success, result.error
        with rasterio.open(output) as src:
            assert src.tags()["FTW_SCENE_ID"] == "scene-1"
            assert src.tags()["FTW_SOURCE"] == "planetscope"
            assert (src.read(1).max(), src.read(2).max()) == (111, 222)

    def test_fetched_workdir_is_removed_after_clipping(self, tmp_path: Path) -> None:
        workdir = tmp_path / ".planet" / "order-1"
        workdir.mkdir(parents=True)
        band_file = _write_band_file(tmp_path / "order.tif")
        source = FakeSource([], {}, band_file=band_file, cleanup=[workdir])

        result = _clip(tmp_path / "out.tif", source)

        assert result.success, result.error
        assert not workdir.exists()

    def test_write_failure_is_reported(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(image_download, "write_cog", lambda *_a, **_k: "disk full")
        source = FakeSource([], {}, band_file=_write_band_file(tmp_path / "order.tif"))

        result = _clip(tmp_path / "out.tif", source)

        assert (result.success, result.error) == (False, "disk full")

    def test_non_child_item_has_no_download_task(self, tmp_path: Path) -> None:
        item = create_mock_stac_item("chip_001", bbox=BBOX)

        with pytest.raises(ValueError, match="not a season child item"):
            download_workflow.build_download_task(item, tmp_path / "chip_001.json")

    def test_child_file_holding_another_item_is_ignored(self, tmp_path: Path) -> None:
        chip_dir = tmp_path / "chips" / "32TNT" / "chip_001"
        chip_dir.mkdir(parents=True)
        write_item(create_mock_stac_item("chip_001"), chip_dir / "chip_001_planting_s2.json")

        assert download_workflow.find_child_items(tmp_path) == []

    @pytest.mark.usefixtures("two_source_chip")
    def test_missing_credentials_stop_the_download(self, tmp_path: Path, monkeypatch) -> None:
        def unavailable(**_kwargs):
            raise SourceUnavailableError("Set PL_API_KEY to use the planetscope source.")

        monkeypatch.setattr(download_workflow, "download_and_clip_scene", unavailable)
        _write_collection(tmp_path)

        with pytest.raises(SourceUnavailableError):
            download_workflow.download_imagery_for_catalog(
                tmp_path, show_progress_bar=False, workers=1, source="planetscope"
            )
        result = CliRunner().invoke(
            cli, ["download-images", str(tmp_path), "--source", "planetscope", "--workers", "1"]
        )

        assert result.exit_code == 1
        assert result.output.count("PL_API_KEY") == 1

    @pytest.mark.usefixtures("two_source_chip")
    def test_cli_counts_pending_orders_as_skipped(self, tmp_path: Path, monkeypatch) -> None:
        def pending(*, scene, output_path, **_kwargs) -> DownloadResult:
            return DownloadResult(
                output_path, scene.id, scene.season, [], 0, 0, "", False, "order running", True
            )

        monkeypatch.setattr(download_workflow, "download_and_clip_scene", pending)
        _write_collection(tmp_path)

        result = CliRunner().invoke(
            cli, ["download-images", str(tmp_path), "--source", "planetscope", "--workers", "1"]
        )

        assert result.exit_code == 0, result.output
        assert "Skipped: 2" in result.output
        assert "Failed: 0" in result.output


def _write_band_file(path: Path) -> Path:
    """A two-band GeoTIFF over BBOX holding 111 and 222."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=20,
        height=20,
        count=2,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_bounds(*BBOX, 20, 20),
    ) as dst:
        dst.write(np.full((20, 20), 111, dtype="uint16"), 1)
        dst.write(np.full((20, 20), 222, dtype="uint16"), 2)
    return path


def _clip(output: Path, source: FakeSource) -> DownloadResult:
    child = create_mock_stac_item(
        "chip_001_planting_planet",
        bbox=BBOX,
        properties={"ftw:source": "planetscope", "ftw:scene_id": "scene-1"},
    )
    scene = SelectedScene(child, "planting", 1.0, child.datetime, "")
    return download_and_clip_scene(
        scene, BBOX, output, ["red", "nir"], resolution=100.0, source=source
    )


class TestPerSourceSelectImagesCli:
    @pytest.fixture(autouse=True)
    def _catalog(self, two_source_chip, tmp_path: Path) -> None:  # noqa: ARG002
        _write_collection(tmp_path)

    def test_clear_selections_only_clears_the_chosen_source(
        self, two_source_chip, tmp_path: Path
    ) -> None:
        chip_dir, _parent = two_source_chip
        unselected = chip_dir.parent / "chip_002"
        unselected.mkdir()
        write_item(create_mock_stac_item("chip_002", bbox=BBOX), unselected / "chip_002.json")
        args = ["select-images", str(tmp_path), "--clear-selections", "--source", "planetscope"]

        cleared = CliRunner().invoke(cli, args, input="y\n")
        again = CliRunner().invoke(cli, args, input="y\n")

        assert cleared.exit_code == 0, cleared.output
        assert "STAC items deleted: 2" in cleared.output
        assert not (chip_dir / "chip_001_planting_planet.json").exists()
        assert (chip_dir / "chip_001_planting_s2.json").exists()
        assert "No chips have planetscope imagery selections to clear." in again.output

    def test_chips_already_selected_for_the_source_are_skipped(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli, ["select-images", str(tmp_path), "--year", "2023", "--source", "planetscope"]
        )

        assert result.exit_code == 0, result.output
        assert "No chips need processing" in result.output
