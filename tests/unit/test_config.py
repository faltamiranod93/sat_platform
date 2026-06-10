# tests/unit/test_config.py
import pytest
import yaml
from pathlib import Path
from satplatform.config import Settings
from satplatform.contracts.geo import CRSRef
from satplatform.composition.di import build_settings

def test_settings_parse_and_paths(tmp_path: Path):
    s = Settings(project_root=tmp_path, crs_out="EPSG:32719")
    assert s.crs_out == "EPSG:32719"
    assert isinstance(s.crs_out_ref(), CRSRef)
    assert s.crs_out_ref().epsg == 32719
    assert s.work_roi_dir.is_absolute()
    p = s.out_path("classmap", date="2025-01-01", classifier="maha")
    assert tmp_path in p.parents

def test_settings_placeholders_guard():
    with pytest.raises(ValueError):
        Settings(output_patterns={"classmap": "artifacts/{site}/x.tif"})

def test_out_path_classmap_layout(tmp_path: Path):
    s = Settings(project_root=tmp_path, crs_out="EPSG:32719")
    out = s.out_path("classmap", date="20240123", classifier="maha")
    assert str(out).replace("\\", "/").endswith("03-Products/CLASSMAP/20240123/classmap_maha.tif")
    vis = s.out_path("classmap_vis", date="20240123", classifier="cos")
    assert str(vis).replace("\\", "/").endswith("03-Products/VIS/20240123/classmap_cos.png")
    summ = s.out_path("compare_summary", name="counts")
    assert str(summ).replace("\\", "/").endswith("04-Analysis/CLASSMAP-COMPARE/counts.csv")

def test_build_settings_from_yaml(tmp_path: Path):
    cfg_dir = tmp_path / "00-Config"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "settings.yaml").write_text(yaml.safe_dump({
        "project_root": str(tmp_path),
        "crs_out": "EPSG:32719",
        "work_roi_dir": "02-Work/ROI",
        "work_products_dir": "03-Products",
        "report_dir": "03-Products/REPORT",
        "gdalwarp_exe": None,
        "band_order": ["B02","B03","B04"],
        "classes": [],
        "input_patterns": {
            "safe_dir": "01-Raw/s2/{product}.SAFE",
            "granule_dir": "01-Raw/s2/{product}.SAFE/GRANULE/{granule}",
            "jp2_file": "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/IMG_DATA/R{res}/T{tile}_{sensing}_{band}_{res}.jp2",
            "scl_file": "01-Raw/s2/{product}.SAFE/GRANULE/{granule}/IMG_DATA/R{res}/T{tile}_{sensing}_SCL_{res}.jp2",
            "roi_file": "00-Config/roi_master.geojson"
        },
        "output_patterns": {
            "classmap": "03-Products/CLASSMAP/{date}/classmap_{classifier}.tif",
            "classmap_vis": "03-Products/VIS/{date}/classmap_{classifier}.png",
            "compare_summary": "04-Analysis/CLASSMAP-COMPARE/{name}.csv"
        }
    }), encoding="utf-8")

    st = build_settings(tmp_path)
    assert st.project_root == tmp_path.resolve()
    out = st.out_path("classmap", date="20250101", classifier="maha")
    assert out.name == "classmap_maha.tif"
    assert "03-Products/CLASSMAP/20250101" in str(out).replace("\\", "/")
