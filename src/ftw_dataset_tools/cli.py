"""Main CLI entry point for FTW Dataset Tools."""

import importlib

import click

from ftw_dataset_tools import __version__

# Command modules are imported on first use to keep startup fast.
COMMAND_MODULES = (
    "add_field_stats",
    "convert_previews",
    "create_boundaries",
    "create_chips",
    "create_dataset",
    "create_dataset_summary",
    "create_ftw_grid",
    "create_masks",
    "create_splits",
    "download_images",
    "get_grid",
    "inspect_fields",
    "run",
    "select_images",
)
_COMMANDS = {module.replace("_", "-"): module for module in COMMAND_MODULES}


class LazyGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *_COMMANDS})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        module = _COMMANDS.get(cmd_name)
        if module is None:
            return super().get_command(ctx, cmd_name)
        return getattr(importlib.import_module(f"ftw_dataset_tools.commands.{module}"), module)


@click.group(cls=LazyGroup)
@click.version_option(version=__version__, prog_name="ftwd")
def cli() -> None:
    """FTW Dataset Tools - CLI for creating Fields of the World benchmark dataset.

    This tool provides commands for:

    \b
    - Creating complete training datasets from field boundaries (create-dataset)
    - Creating FTW grids
    - Creating chip definitions with field coverage statistics
    - Creating boundary lines and raster masks
    """


if __name__ == "__main__":
    cli()
