import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "batch_runner"))


def test_run_monitoring_produces_report(tmp_path, monkeypatch):
    from sklearn.datasets import load_wine

    wine = load_wine()
    columns = [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols",
        "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline", "cluster"
    ]
    df = pd.DataFrame(wine.data, columns=columns[:-1])
    df["cluster"] = 0

    outputs_dir = tmp_path / "wine_clustering" / "outputs"
    outputs_dir.mkdir(parents=True)
    df.to_csv(outputs_dir / "batch_test.csv", index=False)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    monkeypatch.setenv("OUTPUTS_BASE", str(tmp_path))
    monkeypatch.setenv("REPORTS_DIR", str(reports_dir))

    import monitor
    result = monitor.run_monitoring()

    assert "drift_detected" in result
    assert "drift_score" in result
    assert isinstance(result["drift_score"], float)
    reports = list(reports_dir.glob("*.html"))
    assert len(reports) == 1


def test_no_data_returns_safely(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUTS_BASE", str(tmp_path))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))

    import monitor
    result = monitor.run_monitoring()
    assert result is None
