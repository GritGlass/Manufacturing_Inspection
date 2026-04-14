import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_dashboard import configure_page, load_dashboard_data, render_detail_page


configure_page("Detail")
_config, _runs, image_records, _log_entries = load_dashboard_data()
render_detail_page(image_records)
