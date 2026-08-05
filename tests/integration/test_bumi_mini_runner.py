from pathlib import Path

from tools.run_bumi_mini_integration import resolve_source_mini


def test_resolve_source_mini_directory(tmp_path) -> None:
    source = tmp_path / "bundle/source_mini"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text("{}\n", encoding="utf-8")
    assert resolve_source_mini(tmp_path / "bundle", tmp_path / "extract") == source.resolve()
