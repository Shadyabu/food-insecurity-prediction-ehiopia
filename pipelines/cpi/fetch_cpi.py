"""
fetch_cpi.py

Acquisition stage for the headline + food CPI feature. Ethiopia only (see
docs/CPI_Pipeline_Documentation.md Sec 1 for the scope decision -- every
other pipeline in this repo is Ethiopia-only despite Busker et al.'s own
multi-country sourcing, and RQ1 baseline reproduction uses Busker's own
released Horn-of-Africa dataset directly rather than a rebuild here).

TWO independent sources, fetched here:

1. Ha, Kose & Ohnsorge (2021) "One-Stop Source: A Global Database of
   Inflation" (World Bank Policy Research WP 9737) -- the PRIMARY source
   for the raw index level. Confirmed to be a maintained, periodically
   updated product ("updated twice a year" per worldbank.org), not a
   static paper snapshot -- checked 2026-08-09, current vintage "April
   2025" reachable with no auth at the URL below. Provides monthly
   headline (hcpi_m) and food (fcpi_m) CPI INDEX LEVELS for Ethiopia, each
   on the country's own official base period (confirmed distinct base
   periods even within Ethiopia: headline vs food do not share a base --
   this is normal CPI sub-index practice, not a defect -- see the doc).
   As of the April 2025 vintage: Ethiopia headline reaches 2025-03, food
   reaches only 2023-06 (a real, documented recency gap, not a bug).

2. Ethiopian Statistics Service (ESS, ess.gov.et/price/) monthly CPI
   bulletins -- used ONLY to extend the YoY % change (NOT the index
   level -- confirmed 2026-08-09 with the project owner: ESS bulletins
   report inflation RATES only, never a raw index number, so there is no
   level to read off; reconstructing a level via chain-linking off these
   rates is explicitly DEFERRED to a future feature-engineering task, not
   done here -- see docs/CPI_Pipeline_Documentation.md Sec 3) past Ha et
   al.'s coverage, through the present. ESS's own download links use
   irregular WordPress Download Manager slugs (not a predictable URL
   pattern -- checked 2026-08-09, e.g. "cpi_dec_2024" vs
   "consumer-price-index-december-2025" vs
   "1-inflation-report-nov-efy-2017-final" for different months) and the
   actual PDF is served via a `?wpdmdl=<id>` query parameter discovered by
   inspecting each landing page's HTML, not a clean REST endpoint -- so the
   four bulletins below are hardcoded (landing slug, wpdmdl id) pairs
   confirmed reachable at build time, not re-derived from a listing page
   each run.

   Each bulletin's "Table 1: General, Food and Non-Food Inflation Rate
   (%, Year-on-Year)" gives a TRAILING 12-MONTH window ending at the
   bulletin's own issue month, so four bulletins spaced roughly a year
   apart give contiguous Nov-2022-to-present coverage with deliberate
   overlap at two boundaries (Dec-2024, and Jun-2025-to-Dec-2025) for
   internal cross-validation, plus overlap against Ha et al.'s own known
   levels for Nov-2022-to-Jun-2023 (used in validate_cpi_output.py).

SETUP
    pip install requests openpyxl --break-system-packages
"""

import os

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/cpi/ -> pipelines/ -> repo root
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "cpi")
ESS_DIR = os.path.join(RAW_DIR, "ess_bulletins")

HA_ET_AL_URL = (
    "https://thedocs.worldbank.org/en/doc/1ad246272dbbc437c74323719506aa0c-0350012021/"
    "original/Inflation-data.xlsx"
)
HA_ET_AL_PATH = os.path.join(RAW_DIR, "Inflation-data.xlsx")

# (output filename, landing-page slug, wpdmdl id, expected trailing-12-month
# window) -- confirmed reachable 2026-08-09. See docstring above for why
# these can't be derived from a predictable URL pattern.
ESS_BULLETINS = [
    ("cpi_nov_2023.pdf", "cpi_nov_2023", 9151, "Nov 2022 - Nov 2023"),
    ("cpi_dec_2024.pdf", "cpi_dec_2024", 14038, "Dec 2023 - Dec 2024"),
    ("cpi_dec_2025.pdf", "consumer-price-index-december-2025", 17618, "Jan 2025 - Dec 2025"),
    ("cpi_jun_2026.pdf", "consumer-price-index-june-2026", 20016, "Jun 2025 - Jun 2026"),
]

HEADERS = {"User-Agent": "Mozilla/5.0"}


def verify_xlsx_opens(path):
    """Actually parse the cached Excel and check the ETH headline row looks
    plausible -- a truncated/failed download can still "exist" on disk."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if "hcpi_m" not in wb.sheetnames or "fcpi_m" not in wb.sheetnames:
        raise ValueError(f"{path}: missing expected hcpi_m/fcpi_m sheets -- format may have changed")
    ws = wb["hcpi_m"]
    found_eth = any(row[0] == "ETH" for row in ws.iter_rows(values_only=True, max_row=250))
    if not found_eth:
        raise ValueError(f"{path}: no ETH row found in hcpi_m -- file may be truncated or reformatted")
    return True


def verify_pdf_opens(path):
    """Actually parse the cached PDF (not just check it exists) -- a
    partially-downloaded file will still "exist" on disk."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    if len(reader.pages) < 2:
        raise ValueError(f"{path}: only {len(reader.pages)} page(s), expected a multi-page bulletin")
    return True


def _download(url, out_path, verify_fn, max_retries=3):
    if os.path.exists(out_path):
        try:
            verify_fn(out_path)
            print(f"cached file OK: {out_path}")
            return out_path
        except Exception as e:
            print(f"cached file failed verification ({e}), re-downloading")
            os.remove(out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            verify_fn(out_path)
            print(f"downloaded OK: {out_path}")
            return out_path
        except Exception as e:
            if os.path.exists(out_path):
                os.remove(out_path)  # never leave a partial file behind
            print(f"attempt {attempt}/{max_retries} failed for {url}: {e}")
            if attempt == max_retries:
                raise
    return None


def download_ha_et_al():
    return _download(HA_ET_AL_URL, HA_ET_AL_PATH, verify_xlsx_opens)


def download_ess_bulletins():
    paths = []
    for fname, slug, wpdmdl, window in ESS_BULLETINS:
        url = f"https://ess.gov.et/download/{slug}/?wpdmdl={wpdmdl}"
        out_path = os.path.join(ESS_DIR, fname)
        print(f"-- ESS bulletin ({window}) --")
        paths.append(_download(url, out_path, verify_pdf_opens))
    return paths


def main():
    download_ha_et_al()
    download_ess_bulletins()


if __name__ == "__main__":
    main()
