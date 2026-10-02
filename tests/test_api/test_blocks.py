"""Tests for chip block grouping and block hash scores."""

import pandas as pd
import pytest

from ftw_dataset_tools.api.blocks import block_scores, chip_block_ids, mgrs_squares


class TestChipBlockIds:
    def test_explicit_km_size_groups_three_cells(self) -> None:
        ids = pd.Series(["ftw-36NXF0000", "ftw-36NXF0400", "ftw-36NXF0600", "ftw-36NXF0006"])
        assert chip_block_ids(ids, km_size=2).tolist() == [
            "36NXF_0_0",
            "36NXF_0_0",
            "36NXF_1_0",
            "36NXF_0_1",
        ]

    def test_explicit_km_size_is_not_fooled_by_sparse_ids(self) -> None:
        # Eastings 00 and 12 alone infer a step of 12, merging cells 6 blocks apart.
        ids = pd.Series(["ftw-36NXF0000", "ftw-36NXF1200"])
        assert chip_block_ids(ids).nunique() == 1
        assert chip_block_ids(ids, km_size=2).tolist() == ["36NXF_0_0", "36NXF_2_0"]

    def test_rejects_short_ids(self) -> None:
        with pytest.raises(ValueError, match="at least 13 characters"):
            chip_block_ids(pd.Series(["short-id"]))

    def test_rejects_missing_prefix(self) -> None:
        with pytest.raises(ValueError, match="must start with 'ftw-'"):
            chip_block_ids(pd.Series(["xyz-36NXF0000"]))

    def test_rejects_non_numeric_offsets(self) -> None:
        with pytest.raises(ValueError, match="numeric easting/northing"):
            chip_block_ids(pd.Series(["ftw-36NXF00AB"]))


def test_mgrs_squares() -> None:
    assert mgrs_squares(pd.Series(["ftw-36NXF6658"])).tolist() == ["36NXF"]


class TestBlockScores:
    def test_scores_are_pinned(self) -> None:
        # Changing these reshuffles every published scale and split.
        scores = block_scores(pd.Series(["36NXF_0_0", "36NXF_1_0", "36NXF_0_0"]), "ftw-scale-v1")
        assert scores.tolist() == pytest.approx(
            [0.9130356936677305, 0.713448583973568, 0.9130356936677305]
        )
        assert block_scores(pd.Series(["36NXF_0_0"]), "ftw-split-v1").iloc[0] == pytest.approx(
            0.209413951961261
        )

    def test_scores_are_in_unit_interval(self) -> None:
        scores = block_scores(
            pd.Series([f"36NXF_{i}_{j}" for i in range(20) for j in range(20)]), "s"
        )
        assert scores.between(0, 1, inclusive="left").all()
