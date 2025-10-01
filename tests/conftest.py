import os
import pytest
from satplatform.config import get_settings

def pytest_configure():
    os.environ.setdefault("SAT_CRS_OUT", "EPSG:32719")

@pytest.fixture(autouse=True)
def _reset_settings_cache():
    # evita fuga de estado entre tests
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" in item.keywords and os.environ.get("CI") == "true":
            # marca como slow en CI si quieres escalonar
            item.add_marker(pytest.mark.slow)
