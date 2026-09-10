"""
iridl_credentials.py

Loads the IRI Data Library (IRIDL) access key from the git-ignored .env
file at the repo root -- mirrors pipelines/glofas/cds_credentials.py's
pattern exactly (key lives only in .env, never in a tracked file).

IRIDL now requires sign-in for every endpoint (confirmed empirically
2026-08-09 -- every /dods and /data.nc URL 302-redirects to a login page
without a valid session cookie, including this project's own already-
trusted CHIRPS series re-fetched through IRIDL). Auth is cookie-based, not
a request header: generate a key once at
https://iridl.ldeo.columbia.edu/auth/genkey while logged in, then pass it
as the `__dlauth_id` cookie on every request (per IRI's own S2S wiki,
http://wiki.iri.columbia.edu/index.php?n=Climate.S2S-IRIDL). See
docs/IRI_CPC_Seasonal_Pipeline_Documentation.md Sec 9 for the full
registration steps.

.env must define:

    IRI_API_KEY=<your genkey token>
"""

import os

from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/iri_cpc_seasonal/ -> pipelines/ -> repo root
ENV_PATH = os.path.join(REPO_ROOT, ".env")


def get_iridl_cookie():
    """Returns the {'__dlauth_id': <key>} cookie dict for requests.get(),
    loaded from IRI_API_KEY in .env -- never hardcoded, never in a tracked
    file."""
    load_dotenv(ENV_PATH)
    key = os.environ.get("IRI_API_KEY")

    if not key:
        raise RuntimeError(
            f"IRI_API_KEY not found in {ENV_PATH}. Register a free account at "
            "https://iridl.ldeo.columbia.edu, then generate an access key at "
            "https://iridl.ldeo.columbia.edu/auth/genkey and add it to .env as "
            "IRI_API_KEY=<token>. See "
            "docs/IRI_CPC_Seasonal_Pipeline_Documentation.md Sec 9. Never put "
            "the key directly in code or in a tracked file."
        )

    return {"__dlauth_id": key}
