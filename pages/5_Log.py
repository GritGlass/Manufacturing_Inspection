import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_dashboard import configure_page, load_dashboard_data, render_log_page


configure_page("Log")
_config, _runs, _image_records, log_entries = load_dashboard_data()
render_log_page(log_entries)
