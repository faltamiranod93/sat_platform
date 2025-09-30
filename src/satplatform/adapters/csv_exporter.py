## `src/satplatform/adapters/csv_exporter.py`

from __future__ import annotations

import csv
import os
from typing import Mapping, Any, Iterable

from ..ports.exporters import ReportExporterPort

class CSVExporter(ReportExporterPort):
    """Exporter "report" mínimo: escribe un CSV desde `context`.

    Convención:
      - `context["headers"]` -> lista de nombres de columna (opcional)
      - `context["rows"]`    -> iterable de dicts u ordenables
    Si no hay `headers`, se infiere desde la primera fila.
    """
    def render(self, template_id: str, context: Mapping[str, Any], out_uri: str) -> str:
        rows: Iterable[Any] = context.get("rows", [])  # type: ignore[assignment]
        headers = context.get("headers")
        os.makedirs(os.path.dirname(out_uri) or ".", exist_ok=True)
        rows = list(rows)
        if not rows:
            # crea CSV vacío con solo cabecera si se entregó
            with open(out_uri, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if headers: writer.writerow(headers)
            return out_uri
        if headers is None:
            first = rows[0]
            if isinstance(first, Mapping):
                headers = list(first.keys())
            else:
                headers = [f"col{i+1}" for i in range(len(first))]
        with open(out_uri, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                if isinstance(r, Mapping):
                    writer.writerow([r.get(h, "") for h in headers])
                else:
                    writer.writerow(list(r))
        return out_uri
