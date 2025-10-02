# =============================
# FILE: examples/using_catalog_port.py
# =============================
"""
Uso mínimo: CsvCatalog detrás del CatalogPort.
Mantiene tu flujo original pero elimina lecturas sueltas de CSV desde los servicios.
"""
from pathlib import Path
from satplatform.adapters.csv_catalog import CsvCatalog


if __name__ == "__main__":
    root = Path("/ruta/al/proyecto").resolve()
    port = CsvCatalog(project_root=root)

    print("ROIs:")
    for r in port.list_rois():
        print(" -", r.roi_id, r.name, r.geom_path)

    print("Mosaics (primeros 5):")
    for m in port.list_mosaics()[:5]:
        print(" -", m.mosaic_id, m.roi_id, m.acq_date, m.product_path)

    print("Activos (filtrados por fecha):")
    for a in port.iter_assets(date_from=None, date_to=None):
        print(" -", a.asset_path, a.band, a.date)
