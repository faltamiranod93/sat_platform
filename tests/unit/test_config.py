import pytest
from pathlib import Path
from satplatform.config import Settings
from satplatform.contracts.geo import CRSRef

def test_settings_parse_and_paths(tmp_path: Path):
    s = Settings(project_root=tmp_path, crs_out="EPSG:32719")
    assert isinstance(s.crs_out, CRSRef)
    assert s.work_roi_dir.is_absolute()
    p = s.out_path("stack", date="2025-01-01")
    assert tmp_path in p.parents

def test_settings_placeholders_guard():
    with pytest.raises(ValueError):
        Settings(output_patterns={"stack":"artifacts/{site}/x.tif"})
