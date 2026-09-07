"""Capture booth-UI screenshots for the deck.
Drives the real running demo - waits for an actual flagged fraud and its
computed Shapley explanation, so the images show live output, not a loading state.
Run: ./.venv/bin/python deck/capture.py
"""
import time, sys, pathlib
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "http://localhost:8090"
OUT = pathlib.Path(__file__).parent / "img"; OUT.mkdir(exist_ok=True)

opts = Options()
opts.add_argument("--headless")
opts.add_argument("--width=1920"); opts.add_argument("--height=1080")
drv = webdriver.Firefox(service=Service("/snap/bin/geckodriver"), options=opts)
try:
    drv.set_window_size(1920, 1080)
    drv.get(URL)
    W = WebDriverWait(drv, 180)

    print("waiting for the stream to populate ...")
    W.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".tx")) >= 12)

    print("waiting for a flagged fraud ...")
    W.until(lambda d: d.find_elements(By.CSS_SELECTOR, ".tx.fraud"))
    fraud = drv.find_elements(By.CSS_SELECTOR, ".tx.fraud")[0]
    drv.execute_script("arguments[0].click()", fraud)

    print("waiting for the Shapley explanation to render ...")
    W.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#explain .attr")) >= 6)
    time.sleep(1.5)  # let bar widths finish their transition

    # pause the stream so the capture is stable
    drv.execute_script("""
      const e=new KeyboardEvent('keydown',{code:'Space',bubbles:true});
      document.dispatchEvent(e);
    """)
    time.sleep(0.6)

    full = OUT / "demo-full.png"
    drv.save_screenshot(str(full)); print("saved", full)

    # close-up of the explanation panel
    panel = drv.find_element(By.CSS_SELECTOR, "main .panel:nth-child(3)")
    panel.screenshot(str(OUT / "demo-explain.png")); print("saved", OUT/"demo-explain.png")

    # close-up of the verdict / transaction card
    focus = drv.find_element(By.CSS_SELECTOR, "main .panel:nth-child(2)")
    focus.screenshot(str(OUT / "demo-verdict.png")); print("saved", OUT/"demo-verdict.png")

    txt = drv.find_element(By.CSS_SELECTOR, "#focus").text.replace("\n", " | ")
    print("captured transaction:", txt[:160])
finally:
    drv.quit()
