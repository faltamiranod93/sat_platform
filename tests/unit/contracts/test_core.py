import pytest
from datetime import date, datetime
from satplatform.contracts.core import SceneId, CalibrationSpec, RGB8, ClassLabel, MacroClass

def test_sceneid_mgrs_ok():
    s = SceneId(date=date(2025,1,1), tile="19HFE")
    assert s.tile == "19HFE"

@pytest.mark.parametrize("bad", ["61HFE","00HFE","19IOE","19H1E"])
def test_sceneid_mgrs_bad(bad):
    with pytest.raises(ValueError):
        SceneId(date=date(2025,1,1), tile=bad)

def test_calibration_semver_and_checksum():
    c = CalibrationSpec(schema_version="1.2.3", ref_date=date(2024,1,1), checksum="a"*40)
    assert c.schema_version=="1.2.3"

def test_rgb8_and_label():
    col = RGB8(10,20,30)
    lab = ClassLabel(id=1, name="agua", macro=MacroClass.AGUA, color=col)
    assert lab.color.to_hex() == "#0A141E"
