"""Create child STAC items for planting and harvest scenes.

This module provides shared logic for creating child STAC items that is used by both
the standalone `select-images` command and the `create-dataset` pipeline to ensure
identical behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pystac

from ftw_dataset_tools.api.assets import add_file_info, add_raster_bands
from ftw_dataset_tools.api.imagery.naming import (
    SEASONS,
    child_item_id,
    imagery_asset_keys,
    link_source,
    parent_asset_key,
)
from ftw_dataset_tools.api.imagery.sources import (
    DEFAULT_SOURCE,
    SOURCES,
    Sentinel2Source,
    source_class,
)
from ftw_dataset_tools.api.imagery.thumbnails import (
    LEGACY_PREVIEW_MEDIA_TYPE,
    PREVIEW_MEDIA_TYPE,
    PREVIEW_SUFFIX,
)
from ftw_dataset_tools.api.stac import MEDIA_TYPE_COG, _add_portolan_schema
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult, SelectedScene
    from ftw_dataset_tools.api.imagery.sources import ImagerySource

__all__ = [
    "attach_existing_seasons",
    "attach_season_to_parent",
    "attach_thumbnail_to_parent",
    "create_child_items_from_selection",
]

# Links that place an item in the catalog tree; children live next to their parent
# chip item, so the parent's relative hrefs are valid for them verbatim.
_HIERARCHICAL_RELS = ("root", "parent", "collection")


def create_child_items_from_selection(
    chip_dir: Path,
    parent_item: pystac.Item,
    result: SceneSelectionResult,
    year: int,
    cloud_cover_chip: float,
    buffer_days: int,
    num_buffer_expansions: int = 3,
    buffer_expansion_size: int = 14,
) -> None:
    """Create child STAC items for planting and harvest scenes.

    Updates the parent item with FTW properties and creates child items
    with proper links and asset references.

    This is the canonical implementation used by both `select-images` command
    and `create-dataset` pipeline.

    Args:
        chip_dir: Directory containing the chip STAC items
        parent_item: Parent chip STAC item to update
        result: Scene selection result containing planting and harvest scenes
        year: Calendar year for the crop cycle
        cloud_cover_chip: Cloud cover threshold used for selection
        buffer_days: Initial buffer days used for selection
        num_buffer_expansions: Number of buffer expansions configured
        buffer_expansion_size: Days added per expansion

    Side Effects:
        - Updates and saves the parent item with FTW properties
        - Creates and saves child STAC items for planting and harvest
        - Adds ftw:planting and ftw:harvest links to parent item
    """
    # Ensure parent item has self_href set (required for saving with relative links)
    parent_path = chip_dir / f"{parent_item.id}.json"
    if parent_item.get_self_href() is None:
        parent_item.set_self_href(str(parent_path))

    source: ImagerySource = result.source or Sentinel2Source()

    # The crop calendar is shared by every source; the rest of the parent's
    # selection bookkeeping describes the default source's scenes only.
    parent_item.properties["ftw:calendar_year"] = year
    parent_item.properties["ftw:planting_day"] = result.crop_calendar.planting_day
    parent_item.properties["ftw:harvest_day"] = result.crop_calendar.harvest_day
    if source.name == DEFAULT_SOURCE:
        _set_default_source_properties(
            parent_item,
            result,
            cloud_cover_chip=cloud_cover_chip,
            buffer_days=buffer_days,
            num_buffer_expansions=num_buffer_expansions,
            buffer_expansion_size=buffer_expansion_size,
        )

    # Drop this source's previous season links and assets: a rerun that no longer
    # finds a scene for a season must not leave the old one behind.
    stale_rels = {"ftw:planting", "ftw:harvest"}
    if source.name == DEFAULT_SOURCE:
        stale_rels.add("derived")
    parent_item.links = [
        link
        for link in parent_item.links
        if link.rel not in stale_rels
        or (link.rel != "derived" and link_source(link) != source.name)
    ]
    for season in SEASONS:
        parent_item.assets.pop(parent_asset_key(season, "visual", source.name), None)
    for key in imagery_asset_keys(source.name):
        parent_item.assets.pop(key, None)

    # Create each season's child item, then link and mirror it onto the parent.
    # The parent is written in a finally: the FTW selection properties are already
    # set on it, and losing them because one season's child could not be written
    # would leave the chip looking unselected on the next run.
    scenes = {"planting": result.planting_scene, "harvest": result.harvest_scene}
    buffers = {"planting": result.planting_buffer_used, "harvest": result.harvest_buffer_used}
    try:
        for season in SEASONS:
            scene = scenes[season]
            if scene is None:
                continue
            child_item = _create_season_child_item(
                chip_dir=chip_dir,
                parent_item=parent_item,
                scene=scene,
                season=season,
                year=year,
                source=source,
                buffer_used=buffers[season],
            )
            attach_season_to_parent(parent_item, child_item, season, chip_dir=chip_dir)
    finally:
        write_item(parent_item, parent_path)


def _set_default_source_properties(
    parent_item: pystac.Item,
    result: SceneSelectionResult,
    *,
    cloud_cover_chip: float,
    buffer_days: int,
    num_buffer_expansions: int,
    buffer_expansion_size: int,
) -> None:
    properties = parent_item.properties
    properties["ftw:stac_host"] = (result.selection_params or {}).get("stac_host", "earthsearch")
    properties["ftw:cloud_cover_chip_threshold"] = cloud_cover_chip
    properties["ftw:buffer_days"] = buffer_days
    properties["ftw:num_buffer_expansions"] = num_buffer_expansions
    properties["ftw:buffer_expansion_size"] = buffer_expansion_size
    properties["ftw:planting_buffer_used"] = result.planting_buffer_used
    properties["ftw:harvest_buffer_used"] = result.harvest_buffer_used
    properties["ftw:expansions_performed"] = result.expansions_performed

    scenes = [scene for scene in (result.planting_scene, result.harvest_scene) if scene]
    scene_datetimes = [scene.datetime for scene in scenes if scene.datetime]
    if scene_datetimes:
        properties["start_datetime"] = min(scene_datetimes).isoformat()
        properties["end_datetime"] = max(scene_datetimes).isoformat()
    for scene in scenes:
        properties[f"ftw:{scene.season}_cloud_cover"] = round(scene.cloud_cover, 2)


def _provenance(scene: SelectedScene, buffer_used: int | None) -> dict:
    """Where the season's pixels come from and how the chip was rated."""
    properties: dict = {
        "ftw:scene_id": scene.item.id,
        "ftw:cloud_cover_source": scene.cloud_cover_source,
    }
    optional = {
        "ftw:scene_cloud_cover": scene.scene_cloud_cover,
        "ftw:nodata_pct": scene.nodata,
    }
    properties.update({k: round(v, 2) for k, v in optional.items() if v is not None})
    if scene.cloud_mask:
        properties["ftw:cloud_mask"] = scene.cloud_mask
    if buffer_used is not None:
        properties["ftw:buffer_used"] = buffer_used
    if scene.clear_candidates is not None:
        properties["ftw:clear_candidates"] = scene.clear_candidates
    return properties


def _create_season_child_item(
    chip_dir: Path,
    parent_item: pystac.Item,
    scene: SelectedScene,
    season: Literal["planting", "harvest"],
    year: int,
    source: ImagerySource | None = None,
    buffer_used: int | None = None,
) -> pystac.Item:
    """Create and write the child STAC item for one season's scene."""
    source = source or Sentinel2Source()
    child_id = child_item_id(parent_item.id, season, source.name)

    child_path = chip_dir / f"{child_id}.json"
    child_item = pystac.Item(
        id=child_id,
        geometry=parent_item.geometry,
        bbox=parent_item.bbox,
        datetime=scene.datetime,
        properties={
            "ftw:season": season,
            "ftw:source": source.name,
            "ftw:calendar_year": year,
            **_provenance(scene, buffer_used),
            **source.child_properties(scene.item),
        },
    )

    # Set self_href before adding links (required for relative link resolution)
    child_item.set_self_href(str(child_path))

    for key, asset in source.child_assets(scene.item).items():
        child_item.add_asset(key, asset)

    # Add links
    child_item.add_link(
        pystac.Link(
            rel="ftw:parent_chip",
            target=f"./{parent_item.id}.json",
            media_type="application/json",
        )
    )

    if scene.stac_url:
        child_item.add_link(
            pystac.Link(
                rel="via",
                target=scene.stac_url,
                media_type="application/json",
            )
        )

    # Cloud cover over the chip footprint (the item's geometry), not the whole scene
    child_item.properties["eo:cloud_cover"] = round(scene.cloud_cover, 2)

    _copy_hierarchical_links(parent_item, child_item)

    _add_portolan_schema(child_item)

    # Save child item
    write_item(child_item, child_path)

    return child_item


def _copy_hierarchical_links(parent_item: pystac.Item, child_item: pystac.Item) -> None:
    """Give the child the root/parent/collection links its parent chip carries.

    The child item is written into the parent's directory, so the parent's
    relative hrefs address the same catalog and collection from the child. Copies
    the raw hrefs (``transform_href=False``) so an unresolvable root -- normal in
    a staging tree -- is never dereferenced.

    Args:
        parent_item: Parent chip STAC item
        child_item: Child season item to link into the catalog tree
    """
    for rel in _HIERARCHICAL_RELS:
        link = parent_item.get_single_link(rel)
        if link is None:
            continue
        href = link.get_href(transform_href=False)
        if not href:
            continue
        child_item.add_link(pystac.Link(rel=rel, target=href, media_type=link.media_type))

    if parent_item.collection_id:
        child_item.collection_id = parent_item.collection_id


def _scene_id(child_item: pystac.Item) -> str | None:
    """The source scene's id, from ``ftw:scene_id`` or the child's ``via`` link."""
    if child_item.properties.get("ftw:scene_id"):
        return child_item.properties["ftw:scene_id"]
    link = child_item.get_single_link("via")
    href = link.get_href(transform_href=False) if link else None
    if not href:
        return None
    return href.rstrip("/").rsplit("/", 1)[-1] or None


def _attach_visual_asset(
    parent_item: pystac.Item,
    child_item: pystac.Item,
    season: Literal["planting", "harvest"],
    source: str = DEFAULT_SOURCE,
) -> None:
    """Mirror the child's true-colour scene COG onto the parent as ``<season>_visual``.

    The parent's own COGs are label masks and 16-bit band stacks, which render as a
    near-black square in a browser that picks a default asset to draw. This
    ``visual``-role asset gives it a true-colour COG to pick instead.
    """
    visual = child_item.assets.get("visual")
    href = visual.href if visual else None
    if not href or not href.startswith(("http://", "https://")):
        return

    asset = pystac.Asset(
        href=href,
        media_type=MEDIA_TYPE_COG,
        title=f"{season.capitalize()} season scene (true colour)",
        roles=["visual"],
    )
    scene_id = _scene_id(child_item)
    if scene_id:
        asset.extra_fields["ftw:scene"] = scene_id
    scene_datetime = child_item.properties.get("datetime")
    if scene_datetime:
        asset.extra_fields["datetime"] = scene_datetime

    parent_item.add_asset(parent_asset_key(season, "visual", source), asset)


def _band_list_from(asset: pystac.Asset, fallback_title: str | None) -> str | None:
    """The comma-joined band names for an image asset's title, or None.

    Prefers the GeoTIFF's own band descriptions; a file written without them falls
    back to the parenthesised list in the child's ``image`` asset title, which is
    where the download stage writes it (``imagery.image_download`` on the pipeline
    path, ``commands.download_images`` on the standalone one).
    """
    bands = [band.get("description") for band in asset.extra_fields.get("raster:bands", [])]
    named = [band for band in bands if band]
    if named:
        return ",".join(named)
    if fallback_title and "(" in fallback_title and fallback_title.rstrip().endswith(")"):
        return fallback_title.rstrip()[fallback_title.index("(") + 1 : -1] or None
    return None


def _attach_local_image_asset(
    parent_item: pystac.Item,
    child_item: pystac.Item,
    season: Literal["planting", "harvest"],
    chip_dir: Path,
    *,
    source: str = DEFAULT_SOURCE,
    checksums: bool = False,
) -> None:
    """Restore the parent's ``<season>_image`` asset from the child's clipped image.

    Only runs when the clipped GeoTIFF is actually on disk; ``file:size`` and
    ``raster:bands`` are re-read from it rather than copied from the child.
    """
    image = child_item.assets.get("image")
    if image is None:
        return
    filename = Path(image.href).name
    image_path = chip_dir / filename
    if not image_path.exists():
        return

    label = f"{season.capitalize()} season imagery"
    if source != DEFAULT_SOURCE:
        label = f"{season.capitalize()} season {source_class(source).title} imagery"
    asset = pystac.Asset(
        href=f"./{filename}", media_type=MEDIA_TYPE_COG, title=label, roles=["data"]
    )
    parent_item.add_asset(parent_asset_key(season, "image", source), asset)
    add_file_info(asset, image_path, checksum=checksums)
    add_raster_bands(asset, image_path)

    band_list = _band_list_from(asset, image.title)
    if band_list:
        asset.title = f"{label} ({band_list})"


def attach_season_to_parent(
    parent_item: pystac.Item,
    child_item: pystac.Item,
    season: Literal["planting", "harvest"],
    chip_dir: Path | None = None,
    *,
    checksums: bool = False,
) -> None:
    """Link a season's child item to its parent chip and mirror its imagery assets.

    Adds the ``ftw:<season>`` link, the ``<season>_visual`` true-colour scene asset
    and -- when the clipped GeoTIFF is on disk -- the ``<season>_image`` asset.
    Idempotent: a rerun replaces the link and assets rather than duplicating them.

    Args:
        parent_item: Parent chip STAC item to update in place
        child_item: The season child item, freshly created or read back from disk
        season: Season identifier
        chip_dir: Directory holding both items; defaults to the parent's own directory
        checksums: Also compute ``file:checksum`` for the re-attached local image

    Side Effects:
        Mutates ``parent_item``. The caller is responsible for writing it.
    """
    if chip_dir is None:
        self_href = parent_item.get_self_href()
        chip_dir = Path(self_href).parent if self_href else Path()

    source = child_item.properties.get("ftw:source", DEFAULT_SOURCE)
    rel = f"ftw:{season}"
    parent_item.links = [
        link for link in parent_item.links if not (link.rel == rel and link_source(link) == source)
    ]
    parent_item.add_link(
        pystac.Link(
            rel=rel,
            target=f"./{child_item.id}.json",
            media_type="application/json",
            title=f"{season.capitalize()} season {source_class(source).title} imagery",
            extra_fields={"ftw:source": source},
        )
    )

    _attach_visual_asset(parent_item, child_item, season, source)
    _attach_local_image_asset(
        parent_item, child_item, season, chip_dir, source=source, checksums=checksums
    )


#: Chip preview candidates, in the order ``image_download`` itself prefers them:
#: the mask overlay when it could be drawn, otherwise the plain planting preview.
#: Each entry is (filename suffix, media type, asset title).
#:
#: Previews are written as WebP, but the ``.jpg`` entries stay on as a fallback
#: *after* them: a catalog built before that switch still has ``.jpg`` on disk, and
#: its next STAC rerun must keep finding it, or the rerun silently drops the
#: thumbnail asset from every item. Nothing writes a ``.jpg`` preview any more --
#: these entries only read what is already there.
_THUMBNAIL_CANDIDATES = (
    (f"_overlay{PREVIEW_SUFFIX}", PREVIEW_MEDIA_TYPE, "Chip preview with field overlay"),
    (
        f"_planting_image_s2{PREVIEW_SUFFIX}",
        PREVIEW_MEDIA_TYPE,
        "Chip preview (planting season)",
    ),
    ("_overlay.jpg", LEGACY_PREVIEW_MEDIA_TYPE, "Chip preview with field overlay"),
    ("_planting_image_s2.jpg", LEGACY_PREVIEW_MEDIA_TYPE, "Chip preview (planting season)"),
    *(
        (
            f"_planting_image_{cls.suffix}{PREVIEW_SUFFIX}",
            PREVIEW_MEDIA_TYPE,
            "Chip preview (planting season)",
        )
        for name, cls in SOURCES.items()
        if name != DEFAULT_SOURCE
    ),
)


def attach_thumbnail_to_parent(
    parent_item: pystac.Item, chip_dir: Path, *, checksums: bool = False
) -> None:
    """Re-add the chip's preview asset from whichever thumbnail file is on disk.

    The preview is written by the download stage and is not referenced from the
    season children, so a rebuilt chip item would otherwise lose it.

    Args:
        parent_item: Chip item to update in place
        chip_dir: Directory holding the chip's files
        checksums: Also compute ``file:checksum`` for the thumbnail
    """
    for suffix, media_type, title in _THUMBNAIL_CANDIDATES:
        filename = f"{parent_item.id}{suffix}"
        path = chip_dir / filename
        if not path.exists():
            continue
        asset = pystac.Asset(
            href=f"./{filename}",
            media_type=media_type,
            title=title,
            roles=["thumbnail"],
        )
        parent_item.add_asset("thumbnail", asset)
        add_file_info(asset, path, checksum=checksums)
        return


def attach_existing_seasons(
    parent_item: pystac.Item, chip_dir: Path, *, checksums: bool = False
) -> list[str]:
    """Re-attach the season children already on disk to a freshly built chip item.

    The STAC stage rebuilds every chip item from the mask files alone, which would
    otherwise drop the season links and imagery assets that the select/download
    stages added. This reads whichever ``<chip>_<season>_<source>.json`` children
    are present and puts them back, along with the chip's preview thumbnail.

    A child that cannot be parsed -- half-written by an interrupted run, or
    truncated -- is skipped rather than raised: the season is then treated as
    unselected and re-selected on the next pass, which is far cheaper than
    failing catalog generation for the whole dataset.

    Args:
        parent_item: Newly built chip item to update in place
        chip_dir: Directory holding the chip item and its season children
        checksums: Also compute ``file:checksum`` for the re-attached local files

    Returns:
        The seasons that were re-attached for any source, in order.
    """
    attached: list[str] = []
    for season in SEASONS:
        for source in SOURCES:
            child_path = chip_dir / f"{child_item_id(parent_item.id, season, source)}.json"
            if not child_path.exists():
                continue
            try:
                child_item = pystac.Item.from_file(str(child_path))
            except Exception:
                continue
            attach_season_to_parent(
                parent_item, child_item, season, chip_dir=chip_dir, checksums=checksums
            )
            if season not in attached:
                attached.append(season)
    attach_thumbnail_to_parent(parent_item, chip_dir, checksums=checksums)
    return attached
