"""Smoke-test the Streamlit frontend headlessly via Streamlit's AppTest harness.

Skipped automatically if streamlit isn't installed, so the numpy-only host suite still
runs on a minimal environment. When streamlit is present, this runs the actual app
script for every sidebar page and asserts it renders without raising.
"""
import os

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")

PAGES = [
    "Overview",
    "Calibration explorer",
    "Per-class accuracy",
    "Layer sensitivity",
    "DFL plugin validation",
    "Benchmarks",
]


def test_app_default_page_renders():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_each_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"page {page!r} raised: {at.exception}"
