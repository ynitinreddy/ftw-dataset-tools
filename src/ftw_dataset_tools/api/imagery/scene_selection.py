"""Scene selection based on crop calendar dates, for any imagery source."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from ftw_dataset_tools.api.imagery.crop_calendar import (
    CropCalendarDates,
    get_crop_calendar_dates,
)
from ftw_dataset_tools.api.imagery.settings import (
    DEFAULT_BUFFER_DAYS,
    DEFAULT_BUFFER_EXPANSION_SIZE,
    DEFAULT_CLOUD_COVER_CHIP,
    DEFAULT_NODATA_MAX,
    DEFAULT_NUM_BUFFER_EXPANSIONS,
)
from ftw_dataset_tools.api.imagery.sources import Sentinel2Source
from ftw_dataset_tools.api.imagery.sources.base import short_date

if TYPE_CHECKING:
    from collections.abc import Callable

    import pystac

    from ftw_dataset_tools.api.imagery.sources import ImagerySource

__all__ = [
    "SceneSelectionResult",
    "SelectedScene",
    "select_scenes_for_chip",
]


@dataclass
class SelectedScene:
    """A scene chosen for one season of a chip."""

    item: pystac.Item
    season: Literal["planting", "harvest"]
    cloud_cover: float
    datetime: datetime
    stac_url: str
    cloud_cover_source: str = "scene"
    cloud_mask: str | None = None
    nodata: float | None = None
    scene_cloud_cover: float | None = None
    clear_candidates: list[dict] | None = None

    @property
    def id(self) -> str:
        """Scene ID."""
        return self.item.id

    def get_asset_href(self, band: str) -> str | None:
        """Get href for a specific band asset."""
        asset = self.item.assets.get(band)
        return asset.href if asset else None


@dataclass
class SceneSelectionResult:
    """Result of scene selection for a chip."""

    chip_id: str
    bbox: tuple[float, float, float, float]
    year: int
    crop_calendar: CropCalendarDates
    planting_scene: SelectedScene | None = None
    harvest_scene: SelectedScene | None = None
    skipped_reason: str | None = None
    candidates_checked: int = 0
    selection_params: dict = field(default_factory=dict)
    planting_buffer_used: int = 0
    harvest_buffer_used: int = 0
    expansions_performed: int = 0
    source: Any = None

    @property
    def success(self) -> bool:
        """Whether both scenes were successfully selected."""
        return self.planting_scene is not None and self.harvest_scene is not None


def _validate_date_not_future(center_date: datetime, buffer_days: int) -> None:
    """Validate that query dates are not in the future."""
    end_date = center_date + timedelta(days=buffer_days)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if end_date > today:
        raise ValueError(
            f"Query date range extends into the future. "
            f"Center date {center_date.date()} + {buffer_days} buffer days = {end_date.date()}. "
            f"Try using an earlier calendar year."
        )


def _as_aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _get_item_datetime(item: pystac.Item) -> datetime:
    """Timezone-aware acquisition time, falling back to ``start_datetime``."""
    if item.datetime is not None:
        return _as_aware(item.datetime)

    value = item.properties.get("start_datetime") or item.properties.get("datetime")
    if value is None:
        raise ValueError(f"STAC item {item.id} has no datetime or start_datetime")
    if not isinstance(value, datetime | str):
        raise ValueError(f"STAC item {item.id} has invalid start_datetime type: {type(value)}")
    return _as_aware(value)


def _select_best_scene(
    items: list[pystac.Item],
    season: Literal["planting", "harvest"],
    bbox: tuple[float, float, float, float],
    cloud_cover_chip: float = DEFAULT_CLOUD_COVER_CHIP,
    nodata_max: float = DEFAULT_NODATA_MAX,
    on_progress: Callable[[str], None] | None = None,
    source: ImagerySource | None = None,
    record_candidates: bool = False,
) -> SelectedScene | None:
    """The first candidate (best scene-level cloud first) that passes the chip checks.

    With ``record_candidates`` every candidate is checked, and the passing ones are
    listed on the returned scene as ``clear_candidates``.
    """
    source = source or Sentinel2Source()

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    best: SelectedScene | None = None
    candidates: list[dict] = []
    for item in items:
        scene = _assess_candidate(item, season, bbox, cloud_cover_chip, nodata_max, source, log)
        if scene is None:
            continue
        if not record_candidates:
            return scene
        best = best or scene
        candidates.append(
            {
                "scene_id": item.id,
                "datetime": scene.datetime.isoformat(),
                "cloud_cover": round(scene.cloud_cover, 2),
            }
        )

    if best is not None:
        best.clear_candidates = sorted(candidates, key=lambda c: c["datetime"])
    return best


def _assess_candidate(
    item: pystac.Item,
    season: Literal["planting", "harvest"],
    bbox: tuple[float, float, float, float],
    cloud_cover_chip: float,
    nodata_max: float,
    source: ImagerySource,
    log: Callable[[str], None],
) -> SelectedScene | None:
    assessment = source.assess(item, bbox, nodata_max, log)
    if assessment is None:
        return None
    if assessment.cloud_cover > cloud_cover_chip:
        log(f"  Skipping {short_date(item)}: {assessment.cloud_cover:.1f}% cloud")
        return None
    try:
        scene_dt = _get_item_datetime(item)
    except ValueError as e:
        log(f"  {item.id}: skipping - {e}")
        return None
    return SelectedScene(
        item=item,
        season=season,
        cloud_cover=assessment.cloud_cover,
        datetime=scene_dt,
        stac_url=item.get_self_href() or "",
        cloud_cover_source=assessment.cloud_cover_source,
        cloud_mask=assessment.cloud_mask,
        nodata=assessment.nodata,
        scene_cloud_cover=source.scene_cloud_cover(item),
    )


@dataclass
class _SeasonSearch:
    season: Literal["planting", "harvest"]
    center: datetime
    buffer: int
    scene: SelectedScene | None = None
    checked_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Criteria:
    source: ImagerySource
    bbox: tuple[float, float, float, float]
    cloud_cover_chip: float
    nodata_max: float
    record_candidates: bool


def _search_season(
    state: _SeasonSearch, expansion: int, criteria: _Criteria, log: Callable[[str], None]
) -> int:
    """Search one season's window for a scene; returns the new candidates checked."""
    if expansion > 0:
        log(f"Expansion {expansion}: {state.season} buffer now {state.buffer} days")
    log(f"Searching for {state.season} scene around {state.center.date()}...")
    _validate_date_not_future(state.center, state.buffer)

    result = criteria.source.search(criteria.bbox, state.center, state.buffer)
    for line in result.description:
        log(line)

    new_items = [item for item in result.items if item.id not in state.checked_ids]
    if expansion > 0 and len(result.items) > len(new_items):
        log(
            f"Found {len(result.items)} total, {len(new_items)} new {state.season} scene candidates"
        )
    else:
        log(f"Found {len(new_items)} {state.season} scene candidates")
    for item in new_items[:5]:
        cloud = criteria.source.scene_cloud_cover(item)
        log(f"  - {item.id}: {cloud:.1f}% cloud, {item.datetime}")
    state.checked_ids.update(item.id for item in new_items)

    state.scene = _select_best_scene(
        new_items,
        season=state.season,
        bbox=criteria.bbox,
        cloud_cover_chip=criteria.cloud_cover_chip,
        nodata_max=criteria.nodata_max,
        on_progress=log,
        source=criteria.source,
        record_candidates=criteria.record_candidates,
    )
    if state.scene:
        log(
            f"Selected {state.season} scene: {state.scene.id} "
            f"({state.scene.cloud_cover:.1f}% cloud)"
        )
    return len(new_items)


def _skip_reason(planting: SelectedScene | None, harvest: SelectedScene | None) -> str | None:
    if planting is None and harvest is None:
        return "No cloud-free scenes found for either season"
    if planting is None:
        return "No cloud-free planting scene found"
    if harvest is None:
        return "No cloud-free harvest scene found"
    return None


def _finish(result: SceneSelectionResult, seasons: list[_SeasonSearch]) -> SceneSelectionResult:
    planting, harvest = seasons
    result.planting_scene = planting.scene
    result.harvest_scene = harvest.scene
    result.planting_buffer_used = planting.buffer
    result.harvest_buffer_used = harvest.buffer
    if result.skipped_reason is None:
        result.skipped_reason = _skip_reason(planting.scene, harvest.scene)
    return result


def select_scenes_for_chip(
    chip_id: str,
    bbox: tuple[float, float, float, float],
    year: int,
    cloud_cover_chip: float = DEFAULT_CLOUD_COVER_CHIP,
    nodata_max: float = DEFAULT_NODATA_MAX,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    s2_collection: str = "c1",
    num_buffer_expansions: int = DEFAULT_NUM_BUFFER_EXPANSIONS,
    buffer_expansion_size: int = DEFAULT_BUFFER_EXPANSION_SIZE,
    search_backend: str = "parquet",
    on_progress: Callable[[str], None] | None = None,
    source: ImagerySource | None = None,
    record_candidates: bool = False,
) -> SceneSelectionResult:
    """Select a planting and a harvest scene for a chip based on its crop calendar.

    ``source`` defaults to Sentinel-2 built from ``s2_collection`` and
    ``search_backend``. A season that finds no scene has its search window widened
    by ``buffer_expansion_size`` days, up to ``num_buffer_expansions`` times.
    """
    source = source or Sentinel2Source(collection=s2_collection, backend=search_backend)

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    try:
        crop_dates = get_crop_calendar_dates(bbox, on_progress=on_progress)
    except ValueError as e:
        return SceneSelectionResult(
            chip_id=chip_id,
            bbox=bbox,
            year=year,
            crop_calendar=CropCalendarDates(0, 0),
            skipped_reason=f"Crop calendar error: {e}",
            source=source,
        )

    planting_dt, harvest_dt = crop_dates.to_datetime(year)
    log(f"Crop calendar: planting={planting_dt.date()}, harvest={harvest_dt.date()}")

    result = SceneSelectionResult(
        chip_id=chip_id,
        bbox=bbox,
        year=year,
        crop_calendar=crop_dates,
        selection_params={
            "stac_host": source.stac_host,
            "cloud_cover_chip_threshold": cloud_cover_chip,
            "buffer_days": buffer_days,
            "num_buffer_expansions": num_buffer_expansions,
            "buffer_expansion_size": buffer_expansion_size,
        },
        source=source,
    )
    seasons = [
        _SeasonSearch("planting", planting_dt, buffer_days),
        _SeasonSearch("harvest", harvest_dt, buffer_days),
    ]
    criteria = _Criteria(source, bbox, cloud_cover_chip, nodata_max, record_candidates)

    for expansion in range(num_buffer_expansions + 1):
        for state in seasons:
            if state.scene is not None:
                continue
            try:
                result.candidates_checked += _search_season(state, expansion, criteria, log)
            except ValueError as e:
                result.skipped_reason = f"{state.season.capitalize()} query error: {e}"
                return _finish(result, seasons)

        if all(state.scene is not None for state in seasons):
            log(f"Both seasons meet threshold after {expansion} expansion(s)")
            break

        if expansion < num_buffer_expansions:
            for state in seasons:
                if state.scene is None:
                    state.buffer += buffer_expansion_size
            result.expansions_performed = expansion + 1

    return _finish(result, seasons)
