from apex.config import PROJECT_ROOT, load_config, path


def test_config_has_required_sections():
    config = load_config()
    for section in ("paths", "files", "data", "split", "segmentation", "monitoring"):
        assert section in config


def test_target_is_converted():
    assert load_config()["data"]["target"] == "Converted"


def test_path_resolves_inside_project():
    raw_dir = path("raw_dir")
    assert raw_dir.is_absolute()
    assert raw_dir.is_relative_to(PROJECT_ROOT)


def test_segment_shares_are_valid():
    seg = load_config()["segmentation"]
    assert 0 < seg["high_share"] + seg["medium_share"] < 1


def test_psi_thresholds_are_ordered():
    mon = load_config()["monitoring"]
    assert 0 < mon["psi_warning"] < mon["psi_drift"]


def test_bank_config_has_the_same_sections_as_the_lead_config():
    import json

    lead = json.loads((PROJECT_ROOT / "config.json").read_text())
    bank = json.loads((PROJECT_ROOT / "config_bank.json").read_text())
    assert set(bank) == set(lead)
    assert set(bank["data"]) >= set(lead["data"]) - {"category_aliases"}
    assert (
        bank["paths"]["models_dir"] != lead["paths"]["models_dir"]
    )  # never overwrite lead artifacts
    assert bank["paths"]["splits"] != lead["paths"]["splits"]


def test_each_config_lists_leakage_suspects():
    import json

    for name in ("config.json", "config_bank.json"):
        data = json.loads((PROJECT_ROOT / name).read_text())["data"]
        suspects = {c for cols in data["leakage_suspects"].values() for c in cols}
        assert suspects, name
        assert set(data["leakage_columns"]) <= suspects, name
