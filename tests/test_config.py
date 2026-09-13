import yaml

from saferoute.config import load_config


def test_load_base_config():
    cfg = load_config()
    assert cfg["city"]["name"] == "chicago"
    assert isinstance(cfg["spatial"]["h3_resolution"], int)


def test_override_merges_without_losing_base_keys(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text(yaml.dump({"routing": {"default_lambda": 99.0}}))

    cfg = load_config(override_path=override_path)
    assert cfg["routing"]["default_lambda"] == 99.0
    # untouched nested key from base.yaml must survive the merge
    assert cfg["routing"]["default_mu"] == 1.0
    assert cfg["city"]["name"] == "chicago"


def test_path_creates_directory(tmp_path, monkeypatch):
    cfg = load_config()
    p = cfg.path("data_raw")
    assert p.exists()
