"""Tests for imagery sources, per-source catalog layout and scene provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.imagery import catalog_ops, naming, scene_selection
from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates
from ftw_dataset_tools.api.imagery.image_download import DownloadResult, download_and_clip_scene
from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult, SelectedScene
from ftw_dataset_tools.api.imagery.sources import (
    PlanetScopeSource,
    Sentinel2Source,
    SourceUnavailableError,
    build_source,
    planetscope,
)
from ftw_dataset_tools.api.imagery.sources.base import ChipAssessment, FetchResult, SearchResult
from ftw_dataset_tools.api.imagery.stac_child_items import (
    attach_existing_seasons,
    create_child_items_from_selection,
)
from ftw_dataset_tools.api.stac_items import write_item

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


@dataclass
class FakeSource:
    """A source whose candidates and chip ratings are given up front."""

    items: list[pystac.Item]
    cloud: dict[str, float]
    band_file: Path | None = None
    searches: list[int] = field(default_factory=list)

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
            "ready", {band: (str(self.band_file), i + 1) for i, band in enumerate(bands)}
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
        band_file = tmp_path / "order.tif"
        transform = from_bounds(*BBOX, 20, 20)
        with rasterio.open(
            band_file,
            "w",
            driver="GTiff",
            width=20,
            height=20,
            count=2,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(np.full((20, 20), 111, dtype="uint16"), 1)
            dst.write(np.full((20, 20), 222, dtype="uint16"), 2)

        child = create_mock_stac_item(
            "chip_001_planting_planet",
            bbox=BBOX,
            properties={"ftw:source": "planetscope", "ftw:scene_id": "scene-1"},
        )
        scene = SelectedScene(child, "planting", 1.0, child.datetime, "")
        output = tmp_path / "chip_001_planting_image_planet.tif"

        result = download_and_clip_scene(
            scene,
            BBOX,
            output,
            ["red", "nir"],
            resolution=100.0,
            source=FakeSource([], {}, band_file=band_file),
        )

        assert result.success, result.error
        with rasterio.open(output) as src:
            assert src.tags()["FTW_SCENE_ID"] == "scene-1"
            assert src.tags()["FTW_SOURCE"] == "planetscope"
            assert (src.read(1).max(), src.read(2).max()) == (111, 222)
