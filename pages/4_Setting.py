import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_dashboard import configure_page, load_dashboard_data, render_setting_page


configure_page("Setting")
config, runs, _image_records, _log_entries = load_dashboard_data()
latest_run = runs[-1] if runs else None
render_setting_page(config, latest_run)
