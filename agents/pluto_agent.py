"""
pluto_agent.py

Fetches NYC property attributes from the MapPLUTO dataset via the NYC Open
Data Socrata API.  Given a Borough-Block-Lot (BBL) identifier, returns key
structural and valuation attributes used as a third data modality in the
risk assessment pipeline.

PLUTO endpoint (NYC Open Data):
  https://data.cityofnewyork.us/resource/64uk-42ks.json
  Dataset: MapPLUTO (Tax Lot Database)

BBL format: BBOROUGHBLOCKKLOT (10-digit zero-padded string)
  Borough codes: 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island
"""

import os
import requests
from typing import Dict, Optional

PLUTO_ENDPOINT = "https://data.cityofnewyork.us/resource/64uk-42ks.json"
APP_TOKEN = os.environ.get("NYC_OPEN_DATA_TOKEN")  # optional, raises rate limit


# ---------------------------------------------------------------------------
# BBL helpers
# ---------------------------------------------------------------------------

def build_bbl(borough: int, block: str, lot: str) -> str:
    """Return zero-padded 10-digit BBL string."""
    return f"{borough}{block.zfill(5)}{lot.zfill(4)}"


# ---------------------------------------------------------------------------
# Geocode → BBL via NYC Geosupport
# ---------------------------------------------------------------------------

def geocode_to_bbl(address_no: str, street_name: str, borough_code: int) -> Optional[str]:
    """
    Use NYC Planning Geosupport Function 1B to resolve an address to a BBL.
    Returns the 10-character BBL string or None on failure.
    """
    url = "https://geoservice.planning.nyc.gov/geoservice/geoservice.svc/Function_1B"
    params = {
        "AddressNo": address_no,
        "StreetName": street_name,
        "Borough": borough_code,
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Geosupport returns BBL components in the work area
        borough = data.get("LGCsList", [{}])[0].get("BoroughCode1In", "")
        block = data.get("LGCsList", [{}])[0].get("TaxBlock1", "")
        lot = data.get("LGCsList", [{}])[0].get("TaxLot1", "")
        if borough and block and lot:
            return build_bbl(int(borough), block, lot)
        # Fallback: some responses have a flat BBL field
        bbl = data.get("BBLE") or data.get("BBL")
        return str(bbl)[:10] if bbl else None
    except Exception as e:
        print(f"[PLUTO] Geocode-to-BBL failed: {e}")
        return None


# ---------------------------------------------------------------------------
# PLUTO lookup
# ---------------------------------------------------------------------------

def get_property_data(bbl: str) -> Dict:
    """
    Fetch MapPLUTO record for the given BBL.

    Returns a dict with selected attributes relevant to risk assessment:
      bldgclass, yearbuilt, numfloors, landuse, assesstot, lotarea,
      unitsres, bldgarea, zonedist1, histdist
    """
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    params = {"bbl": bbl, "$limit": 1}
    try:
        r = requests.get(PLUTO_ENDPOINT, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return {"error": f"No PLUTO record found for BBL {bbl}"}
        row = rows[0]
        return {
            "bbl": bbl,
            "bldgclass": row.get("bldgclass"),       # Building class code
            "yearbuilt": row.get("yearbuilt"),        # Year built
            "numfloors": row.get("numfloors"),        # Number of floors
            "landuse": row.get("landuse"),            # Land use code
            "assesstot": row.get("assesstot"),        # Total assessed value ($)
            "lotarea": row.get("lotarea"),            # Lot area (sq ft)
            "unitsres": row.get("unitsres"),          # Residential units
            "bldgarea": row.get("bldgarea"),          # Gross floor area (sq ft)
            "zonedist1": row.get("zonedist1"),        # Primary zoning district
            "histdist": row.get("histdist"),          # Historic district (if any)
            "ownertype": row.get("ownertype"),        # Owner type
        }
    except Exception as e:
        return {"error": str(e)}


def get_property_data_from_address(
    address_no: str, street_name: str, borough_code: int
) -> Dict:
    """
    Query PLUTO directly by address string and borough code.
    Uses a LIKE filter so minor street-name variations still match.
    """
    BORO_CODE_MAP = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    borocode = BORO_CODE_MAP.get(int(borough_code))

    # Normalize: uppercase, collapse spaces
    street_words = street_name.upper().split()
    # Build a LIKE pattern from the first meaningful word of the street name
    street_keyword = street_words[0] if street_words else street_name.upper()
    where = (
        f"address like '{address_no}%{street_keyword}%' "
        f"AND borocode='{borocode}'"
    )
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    params = {"$where": where, "$limit": 1}
    try:
        r = requests.get(PLUTO_ENDPOINT, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return {"error": f"No PLUTO record found for {address_no} {street_name}"}
        row = rows[0]
        bbl = f"{borocode}{row.get('block', '').zfill(5)}{row.get('lot', '').zfill(4)}"
        return {
            "bbl": bbl,
            "address": row.get("address"),
            "bldgclass": row.get("bldgclass"),
            "yearbuilt": row.get("yearbuilt"),
            "numfloors": row.get("numfloors"),
            "landuse": row.get("landuse"),
            "assesstot": row.get("assesstot"),
            "lotarea": row.get("lotarea"),
            "unitsres": row.get("unitsres"),
            "bldgarea": row.get("bldgarea"),
            "zonedist1": row.get("zonedist1"),
            "histdist": row.get("histdist"),
            "ownertype": row.get("ownertype"),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 621 Morgan Avenue, Brooklyn → Borough 3
    # Known BBL for a building on Morgan Ave: 3-02613-0001 → "3026130001"
    test_bbl = "3026130001"
    result = get_property_data(test_bbl)

    print("\n--- PLUTO Property Record ---")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        for k, v in result.items():
            print(f"  {k:15s}: {v}")

    print("\n--- From Address ---")
    result2 = get_property_data_from_address("621", "Morgan Avenue", 3)
    if "error" in result2:
        print(f"Error: {result2['error']}")
    else:
        for k, v in result2.items():
            print(f"  {k:15s}: {v}")
