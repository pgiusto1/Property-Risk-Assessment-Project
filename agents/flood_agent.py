import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

# --- Load census tracts and FVI data ---
TRACTS_PATH = "data/nyct2020_25a/nyct2020.shp"
FVI_CSV_PATH = "data/New_York_City_s_Flood_Vulnerability_Index.csv"

tracts = gpd.read_file(TRACTS_PATH).to_crs("EPSG:4326")
fvi = pd.read_csv(FVI_CSV_PATH)

# Ensure GEOID is string for merging
tracts["GEOID"] = tracts["GEOID"].astype(str)
fvi["GEOID"] = fvi["GEOID"].astype(str)

# Merge FVI into tract geometries
merged = tracts.merge(fvi, on="GEOID")

BOROUGH_NAMES = {
    1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"
}


# --- Geocode via Nominatim (OpenStreetMap) ---
def geocode_address_1b(address_no, street_name, borough_code):
    """Return (lat, lon) for an NYC address using the Nominatim geocoder."""
    borough_name = BOROUGH_NAMES.get(int(borough_code), "New York")
    query = f"{address_no} {street_name}, {borough_name}, NY"
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "nyc-property-risk-assessment/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        results = r.json()
        if not results:
            print(f"Nominatim: no results for '{query}'")
            return None
        lat = float(results[0]["lat"])
        lon = float(results[0]["lon"])
        return lat, lon
    except Exception as e:
        print(f"Geocoding failed: {e}")
        return None

# --- Get vulnerability by address ---
def get_flood_vulnerability_from_address(address_no, street_name, borough_code):
    coords = geocode_address_1b(address_no, street_name, borough_code)
    if coords is None:
        return {"error": "Address could not be geocoded"}
    return get_flood_vulnerability(*coords)

# --- Get vulnerability by lat/lon ---
def get_flood_vulnerability(lat, lon):
    point = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"
    )
    match = gpd.sjoin(point, merged, how="left", predicate="intersects")
    if match.empty:
        return {"flood_risk": "No data", "FVI": None}

    row = match.iloc[0]
    return {
        "GEOID": row["GEOID"],
        "FVI_tidal_2020s": row.get("FVI_tidal_2020s"),
        "FVI_tidal_2050s": row.get("FVI_tidal_2050s"),
        "FVI_tidal_2080s": row.get("FVI_tidal_2080s"),
        "FVI_storm_surge_present": row.get("FVI_storm_surge_present"),
        "FVI_storm_surge_2050s": row.get("FVI_storm_surge_2050s"),
        "FVI_storm_surge_2080s": row.get("FVI_storm_surge_2080s"),
        "FSHRI": row.get("FSHRI"),
    }

# --- Test ---
if __name__ == "__main__":
    # "621 Morgan Avenue, Brooklyn" → 621, "Morgan Avenue", 3
    result = get_flood_vulnerability_from_address("621", "Morgan Avenue", 3)

    print("\n--- Flood Risk Assessment ---")
    if "error" in result:
        print(result["error"])
    else:
        for k, v in result.items():
            print(f"{k}: {v}")
