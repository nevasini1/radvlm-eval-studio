"""Internal helper: drive the running Streamlit app and capture tab screenshots.

Not part of the shipped package — used once to populate docs/screenshots/.
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def wait_idle(page, ms=1500):
    """Wait for Streamlit's run to settle."""
    page.wait_for_timeout(ms)
    try:
        # Status widget ("RUNNING") disappears when idle.
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=4000)
    except Exception:
        pass
    page.wait_for_timeout(400)


def click_tab(page, name):
    page.get_by_role("tab", name=name).click()
    wait_idle(page)


def shot(page, filename):
    path = OUT / filename
    page.screenshot(path=str(path), full_page=True)
    print(f"saved {path}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1480, "height": 1000}, device_scale_factor=2)
        page.goto(URL, wait_until="load")
        # Wait for the app to render.
        page.wait_for_selector("text=RadVLM Eval Studio", timeout=30000)
        wait_idle(page, 2500)

        # 1) Overview (default tab)
        shot(page, "overview.png")

        # 2) Study Viewer (default study)
        click_tab(page, "Study Viewer")
        shot(page, "study_viewer.png")

        # 3) Similar Cases
        click_tab(page, "Similar Cases")
        shot(page, "similar_cases.png")

        # Select a pneumothorax study (DEMO0005) so the draft/eval are interesting.
        try:
            page.locator('[data-testid="stSelectbox"]').first.click()
            wait_idle(page, 600)
            page.get_by_role("option", name="DEMO0005", exact=True).click()
            wait_idle(page, 1500)
        except Exception as exc:
            print(f"study-select skipped: {exc}")

        # 4) Draft Report -> load seeded flawed draft
        click_tab(page, "Draft Report")
        try:
            page.get_by_role("button", name="Load seeded flawed draft").click()
            wait_idle(page, 1500)
        except Exception as exc:
            print(f"seed-draft skipped: {exc}")
        shot(page, "draft_report.png")

        # 5) Evaluation (should be RED for the seeded pneumothorax draft)
        click_tab(page, "Evaluation")
        shot(page, "evaluation.png")

        # 6) Review & Audit Log -> perform an edit so before/after + audit show
        click_tab(page, "Review & Audit Log")
        try:
            ta = page.locator('textarea').first
            ta.click()
            wait_idle(page, 300)
            page.get_by_role("button", name="Save edit").click()
            wait_idle(page, 1500)
        except Exception as exc:
            print(f"review-action skipped: {exc}")
        shot(page, "review.png")

        browser.close()


if __name__ == "__main__":
    main()
