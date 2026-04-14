from __future__ import annotations

from scripts.utils import configure_page, load_dashboard_data, render_home_page


def main() -> None:
    configure_page("Dashboard Home")
    config, runs, _image_records, log_entries = load_dashboard_data()
    render_home_page(config, runs, log_entries)


if __name__ == "__main__":
    main()
