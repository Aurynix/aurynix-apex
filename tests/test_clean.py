import pandas as pd
import pytest

from apex.config import load_config
from apex.data.clean import (
    apply_aliases,
    cap_outliers,
    clean,
    encode_binary,
    fill_missing,
    group_other,
    normalize_text,
    replace_placeholders,
    validate,
)


@pytest.fixture
def raw():
    """A few raw leads with the issues found in Leads.csv."""
    return pd.DataFrame(
        {
            "Prospect ID": ["a", "b", "c", "d"],
            "Lead Number": [1, 2, 3, 4],
            "Lead Source": ["Google", "google", " Olark Chat ", None],
            "Do Not Email": ["No", "Yes", "No", "No"],
            "Magazine": ["No", "No", "No", "No"],
            "Converted": [1, 0, 0, 1],
            "TotalVisits": [5.0, 0.0, None, 2.0],
            "City": ["Mumbai", "Select", None, "Select"],
            "Country": ["India", "unknown", "India", None],
            "Tags": ["Interested  in full time MBA", None, "Busy", "Busy"],
        }
    )


def test_normalize_text_strips_and_collapses_spaces(raw):
    out = normalize_text(raw)
    assert out.loc[2, "Lead Source"] == "Olark Chat"
    assert out.loc[0, "Tags"] == "Interested in full time MBA"
    assert pd.isna(out.loc[3, "Lead Source"])


def test_replace_placeholders_turns_select_into_missing(raw):
    out = replace_placeholders(raw, ["Select", "unknown"])
    assert out["City"].isna().sum() == 3
    assert pd.isna(out.loc[1, "Country"])
    assert out.loc[0, "City"] == "Mumbai"


def test_apply_aliases_merges_spelling_variants(raw):
    out = apply_aliases(raw, {"Lead Source": {"google": "Google"}})
    assert (out["Lead Source"] == "Google").sum() == 2


def test_apply_aliases_ignores_missing_columns(raw):
    out = apply_aliases(raw, {"Not A Column": {"x": "y"}})
    pd.testing.assert_frame_equal(out, raw)


def test_encode_binary_maps_yes_no(raw):
    out = encode_binary(raw, ["Do Not Email"])
    assert out["Do Not Email"].tolist() == [0, 1, 0, 0]


def test_encode_binary_rejects_unexpected_values(raw):
    raw.loc[0, "Do Not Email"] = "Maybe"
    with pytest.raises(ValueError, match="Maybe"):
        encode_binary(raw, ["Do Not Email"])


def test_validate_rejects_bad_target(raw):
    raw.loc[0, "Converted"] = 2
    with pytest.raises(ValueError, match="Converted"):
        validate(raw, "Converted", [])


def test_validate_rejects_negative_numbers(raw):
    raw.loc[0, "TotalVisits"] = -1
    with pytest.raises(ValueError, match="TotalVisits"):
        validate(raw, "Converted", ["TotalVisits"])


def test_validate_rejects_page_views_without_visits():
    df = pd.DataFrame({"TotalVisits": [0], "Page Views Per Visit": [2.0]})
    with pytest.raises(ValueError, match="0 visits"):
        validate(df, None, [])


def test_group_other_keeps_listed_values_and_missing():
    df = pd.DataFrame({"Country": ["India", "Qatar", None]})
    out = group_other(df, {"Country": ["India"]})
    assert out["Country"].tolist()[:2] == ["India", "Other"]
    assert pd.isna(out.loc[2, "Country"])


def test_fill_missing_uses_fixed_values():
    df = pd.DataFrame({"TotalVisits": [None, 4.0], "City": [None, "Mumbai"]})
    out = fill_missing(df, {"TotalVisits": 3}, "Missing")
    assert out["TotalVisits"].tolist() == [3, 4]
    assert out["City"].tolist() == ["Missing", "Mumbai"]


def test_cap_outliers_clips_but_keeps_rows():
    df = pd.DataFrame({"TotalVisits": [2, 251]})
    out = cap_outliers(df, {"TotalVisits": 30})
    assert out["TotalVisits"].tolist() == [2, 30]


def test_clean_end_to_end(raw):
    out = clean(raw)
    # IDs and configured constant columns are dropped
    assert not {"Prospect ID", "Lead Number", "Magazine"} & set(out.columns)
    # rows are never dropped: missing values carry signal (docs/data_cleaning.md)
    assert len(out) == len(raw)
    assert (out["Lead Source"] == "Google").sum() == 2
    assert (out["City"] == "Missing").sum() == 3
    assert out.drop(columns=["Converted"]).notna().all().all()
    assert out["TotalVisits"].dtype == int


def test_clean_works_without_target(raw):
    """New leads at prediction time have no label."""
    out = clean(raw.drop(columns=["Converted"]))
    assert "Converted" not in out.columns


def test_clean_drops_configured_leakage_columns(raw):
    config = load_config()
    config = {**config, "data": {**config["data"], "leakage_columns": ["Tags"]}}
    assert "Tags" not in clean(raw, config).columns


def test_clean_is_stateless(raw):
    """Cleaning one lead gives the same result as cleaning it within a batch."""
    batch = clean(raw)
    single = clean(raw.iloc[[1]])
    pd.testing.assert_frame_equal(
        single.reset_index(drop=True), batch.iloc[[1]].reset_index(drop=True)
    )
