"""
cds_credentials.py

Loads Copernicus CDS/EWDS credentials from a git-ignored .env file at the
repo root, rather than the cdsapi library's default of reading
~/.cdsapirc. Keeping the key in .env (never committed -- see .gitignore)
means it never ends up inside this repo's tracked files, unlike a
.cdsapirc some contributors might otherwise be tempted to place in-repo
"for convenience."

.env is not read automatically by anything except this helper -- it is
loaded explicitly via python-dotenv, and the two variables it must define
are:

    CDS_API_URL=https://ewds.climate.copernicus.eu/api
    CDS_API_KEY=<your personal access token>

See docs/GloFAS_Pipeline_Documentation.md Sec 9 for how to obtain these.
"""

import os

from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ENV_PATH = os.path.join(REPO_ROOT, ".env")


def get_cds_client():
    """Returns a cdsapi.Client() constructed from CDS_API_URL/CDS_API_KEY
    in .env -- never from ~/.cdsapirc, so the key's only on-disk location
    is the git-ignored .env file."""
    import cdsapi  # imported here, not at module top, so this file can be
                     # imported without cdsapi installed (e.g. for a quick
                     # credentials-presence check)

    load_dotenv(ENV_PATH)
    url = os.environ.get("CDS_API_URL")
    key = os.environ.get("CDS_API_KEY")

    if not url or not key:
        raise RuntimeError(
            f"CDS_API_URL / CDS_API_KEY not found in {ENV_PATH}. "
            "Copy .env.example to .env at the repo root and fill in your "
            "Copernicus CDS/EWDS personal access token -- see "
            "docs/GloFAS_Pipeline_Documentation.md Sec 9 for how to get one. "
            "Never put the key directly in code or in a tracked file."
        )

    return cdsapi.Client(url=url, key=key)
