"""Image download orchestration for STAC catalogs.

This module provides workflow functions for downloading imagery for all child
items in a catalog. It is used by both the standalone `download-images` command
and the `create-dataset` pipeline to ensure identical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs
from ftw_dataset_tools.api.imagery.image_download import (
    download_and_clip_scene,
    process_downloaded_scene,
)
from ftw_dataset_tools.api.imagery.naming import image_filename, parse_child_id
from ftw_dataset_tools.api.imagery.parallel import (
    DEFAULT_WORKERS,
    ParallelOutcome,
    run_in_parallel,
)
from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene
from ftw_dataset_tools.api.imagery.sources import (
    DEFAULT_SOURCE,
    SourceUnavailableError,
    build_source,
)
from ftw_dataset_tools.api.imagery.thumbnails import has_rgb_bands

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.image_download import DownloadResult
    from ftw_dataset_tools.api.imagery.sources import ImagerySource

__all__ = [
    "DownloadTask",
    "DownloadWorkflowResult",
    "SourceCache",
    "build_download_task",
    "download_imagery_for_catalog",
    "download_task_scene",
    "find_child_items",
    "find_s2_child_items",
    "skip_download_reason",
]

DEFAULT_BANDS = ("red", "green", "blue", "nir")


@dataclass
class DownloadWorkflowResult:
    """Result of running image download across a catalog."""

    successful: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    failed_details: list[dict] = field(default_factory=list)


def find_child_items(
    catalog_dir: Path,
    unreadable: list[dict] | None = None,
    source: str | None = None,
) -> list[tuple[pystac.Item, Path]]:
    """Find the season child items in a catalog directory, for one source or all.

    Args:
        catalog_dir: Path to the collection directory (holding collection.json),
                     whose ``chips/<square>/<item_id>/`` subdirectories hold STAC
                     item files
        unreadable: Optional list that collects ``{"item": ..., "error": ...}``
                    entries for item files that could not be parsed. An item that
                    cannot be read is otherwise invisible: it is not downloaded and
                    is counted nowhere, so pass this in to report it.
        source: Only this source's children (default: every source)

    Returns:
        List of (pystac.Item, item_path) tuples for each child item found.
    """
    child_items = []

    for subdir in iter_chip_dirs(catalog_dir):
        for json_file in sorted(subdir.glob("*.json")):
            ref = parse_child_id(json_file.stem)
            if ref is None or (source is not None and ref.source != source):
                continue
            try:
                item = pystac.Item.from_file(str(json_file))
            except Exception as e:
                if unreadable is not None:
                    unreadable.append({"item": json_file.stem, "error": f"Unreadable item: {e}"})
                continue
            if parse_child_id(item.id) is not None:
                child_items.append((item, json_file))

    return child_items


def find_s2_child_items(
    catalog_dir: Path,
    unreadable: list[dict] | None = None,
) -> list[tuple[pystac.Item, Path]]:
    """Find the Sentinel-2 season child items in a catalog directory."""
    return find_child_items(catalog_dir, unreadable, source=DEFAULT_SOURCE)


class SourceCache:
    """One configured source per name, built on first use."""

    def __init__(self, source: ImagerySource | None = None, **options: object) -> None:
        self._options = options
        self._sources: dict[str, ImagerySource] = {source.name: source} if source else {}

    def get(self, name: str) -> ImagerySource:
        if name not in self._sources:
            self._sources[name] = build_source(name, **self._options)  # type: ignore[arg-type]
        return self._sources[name]


def download_imagery_for_catalog(
    catalog_dir: Path,
    bands: list[str] | None = None,
    resolution: float = 10.0,
    generate_thumbnails: bool = True,
    resume: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    show_progress_bar: bool = True,
    workers: int = DEFAULT_WORKERS,
    source: str | None = None,
    source_options: dict | None = None,
) -> DownloadWorkflowResult:
    """Download imagery for all season child items in a catalog.

    This is the core orchestration function used by both `download-images` command
    and `create-dataset` pipeline.

    Scenes are fetched on a thread pool; the STAC writes that follow each download
    stay on the calling thread, since the two child items of one chip update the
    same parent item file.

    Args:
        catalog_dir: Path to the chips collection directory
        bands: List of bands to download. Default: ["red", "green", "blue", "nir"]
        resolution: Target resolution in meters
        generate_thumbnails: Whether to generate WebP preview thumbnails
        resume: Skip items that already have local imagery. Defaults to True: a
            completed download replaces the child's band assets with the local
            `image`, so re-attempting one leaves no band hrefs to fetch and fails.
            Pass False to force a re-download, which is what a changed `bands` or
            `resolution` needs - and which needs the remote band refs back first.
        on_progress: Optional callback (current, total) for progress updates
        show_progress_bar: If True, show tqdm progress bar
        workers: Number of scenes to download concurrently
        source: Only download this source's children (default: every source)
        source_options: Keyword options for :func:`build_source` (e.g. Planet settings)

    Returns:
        DownloadWorkflowResult with success/skipped/failed counts and details
    """
    band_list = list(bands) if bands is not None else list(DEFAULT_BANDS)
    result = DownloadWorkflowResult()
    can_generate_thumbnail = generate_thumbnails and has_rgb_bands(band_list)
    sources = SourceCache(**(source_options or {}))

    # Items whose JSON cannot be read are reported as failures rather than
    # silently dropped from the run.
    unreadable: list[dict] = []
    child_items = find_child_items(catalog_dir, unreadable=unreadable, source=source)
    result.failed += len(unreadable)
    result.failed_details.extend(unreadable)

    if not child_items:
        return result

    progress_bar = (
        tqdm(total=len(child_items), desc="Downloading imagery", unit="scene", leave=False)
        if show_progress_bar
        else None
    )
    done = 0

    def advance() -> None:
        nonlocal done
        done += 1
        if progress_bar:
            progress_bar.update(1)
        if on_progress:
            on_progress(done, len(child_items))

    try:
        tasks: list[DownloadTask] = []
        for item, item_path in child_items:
            skip_reason = skip_download_reason(item, item_path, resume=resume)
            if skip_reason is not None:
                result.skipped += 1
                result.skipped_details.append({"item": item.id, "reason": skip_reason})
                advance()
                continue
            tasks.append(build_download_task(item, item_path))
        resolved = {task.source: sources.get(task.source) for task in tasks}

        def work(task: DownloadTask) -> DownloadResult:
            return download_task_scene(
                task, bands=band_list, resolution=resolution, source=resolved[task.source]
            )

        def apply(outcome: ParallelOutcome[DownloadTask, DownloadResult]) -> None:
            _record_download(
                outcome,
                result=result,
                band_list=band_list,
                generate_thumbnails=can_generate_thumbnail,
            )
            advance()

        run_in_parallel(tasks, work=work, apply=apply, workers=workers)

    finally:
        if progress_bar:
            progress_bar.close()

    return result


@dataclass
class DownloadTask:
    """One scene to fetch, with the paths its outputs go to.

    ``logs`` collects the download's progress lines so the caller can replay them
    in one block rather than have concurrent downloads interleave their output.
    """

    item: pystac.Item
    item_path: Path
    bbox: tuple[float, ...]
    season: Literal["planting", "harvest"]
    base_id: str
    output_filename: str
    output_path: Path
    logs: list[str] = field(default_factory=list)
    source: str = DEFAULT_SOURCE


def skip_download_reason(item: pystac.Item, item_path: Path, *, resume: bool) -> str | None:
    """Return why this item needs no download, or None if it does."""
    if not item.bbox:
        return "No bbox in item"

    if not resume:
        return None

    local_asset = item.assets.get("image") or item.assets.get("clipped")
    if local_asset is None:
        return None

    local_path = item_path.parent / local_asset.href.lstrip("./")
    return "Already downloaded" if local_path.exists() else None


def build_download_task(item: pystac.Item, item_path: Path) -> DownloadTask:
    """Derive the season, source and output paths for one child item."""
    ref = parse_child_id(item.id)
    if ref is None:
        raise ValueError(f"{item.id} is not a season child item")
    output_filename = image_filename(ref.chip_id, ref.season, ref.source)

    return DownloadTask(
        item=item,
        item_path=item_path,
        bbox=tuple(item.bbox),
        season=ref.season,
        base_id=ref.chip_id,
        output_filename=output_filename,
        output_path=item_path.parent / output_filename,
        source=ref.source,
    )


def download_task_scene(
    task: DownloadTask,
    *,
    bands: list[str],
    resolution: float,
    source: ImagerySource | None = None,
) -> DownloadResult:
    """Fetch and clip one scene.

    Safe to run on a worker thread: it writes only this task's own GeoTIFF and
    collects its log lines instead of printing them.
    """
    scene = SelectedScene(
        item=task.item,
        season=task.season,
        cloud_cover=task.item.properties.get("eo:cloud_cover", 0.0),
        datetime=task.item.datetime,
        stac_url=task.item.get_self_href() or "",
    )

    return download_and_clip_scene(
        scene=scene,
        bbox=task.bbox,
        output_path=task.output_path,
        bands=bands,
        resolution=resolution,
        on_progress=task.logs.append,
        source=source or build_source(task.source),
        child_path=task.item_path,
    )


def _record_download(
    outcome: ParallelOutcome[DownloadTask, DownloadResult],
    *,
    result: DownloadWorkflowResult,
    band_list: list[str],
    generate_thumbnails: bool,
) -> None:
    """Update the STAC items for one finished download (calling thread only).

    The parent chip item is shared between a chip's planting and harvest items,
    so this must never run concurrently with itself.
    """
    task = outcome.task

    if isinstance(outcome.error, SourceUnavailableError):
        raise outcome.error
    if outcome.error is not None:
        result.failed += 1
        result.failed_details.append({"item": task.item.id, "error": str(outcome.error)})
        return

    download_result = outcome.value
    if download_result is not None and download_result.pending is True:
        result.skipped += 1
        result.skipped_details.append({"item": task.item.id, "reason": download_result.error})
        return
    if download_result is None or not download_result.success:
        error = download_result.error if download_result else None
        result.failed += 1
        result.failed_details.append({"item": task.item.id, "error": error or "Unknown error"})
        return

    try:
        process_downloaded_scene(
            item=task.item,
            item_path=task.item_path,
            output_path=task.output_path,
            output_filename=task.output_filename,
            band_list=band_list,
            season=task.season,
            base_id=task.base_id,
            generate_thumbnails=generate_thumbnails,
        )
    except Exception as err:  # reported per item, like a failed download
        result.failed += 1
        result.failed_details.append({"item": task.item.id, "error": str(err)})
        return

    result.successful += 1
