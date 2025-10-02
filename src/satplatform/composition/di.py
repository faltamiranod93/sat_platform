from __future__ import annotations
from pathlib import Path
import json
import yaml

from ..config import Settings
from ..contracts.core import ClassLabel, MacroClass, RGB8

def load_settings_from_yaml(path: Path) -> Settings:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(**data)

def load_class_labels(path: Path) -> tuple[ClassLabel, ...]:
    items = json.loads(path.read_text(encoding="utf-8"))
    out: list[ClassLabel] = []
    for it in items:
        out.append(
            ClassLabel(
                id=int(it["id"]),
                name=str(it["name"]),
                macro=MacroClass(it["macro"]),
                color=RGB8(**it.get("color", {})),
            )
        )
    return tuple(out)

def build_settings(project_root: Path) -> Settings:
    cfg = (project_root / "00-Config" / "settings.yaml").resolve()
    st = load_settings_from_yaml(cfg)
    if not st.classes:
        labels_json = (project_root / "00-Config" / "class_labels.json").resolve()
        if labels_json.exists():
            st = st.model_copy(update={"classes": load_class_labels(labels_json)})
    return st
