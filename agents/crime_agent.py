#!/usr/bin/env python3
"""
crime_agent.py

Fetches NYC crime severity by sampling high-density points on a grid.
Selects 35 grid points within the borough that each have ≥100 complaints
in a 0.25 mile radius, then computes the mean and standard deviation
of their severity scores. Also computes severity at a specific address: 621 Morgan Ave, Greenpoint.
"""
import random
import requests
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Constants
SODA_ENDPOINT   = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
API_LIMIT        = 1000
SEVERITY_WEIGHTS = {"FELONY":3, "MISDEMEANOR":2, "VIOLATION":1, "UNKNOWN":0}
EPSG_NYC_UTM     = 32618  # UTM Zone 18N (meters)
YEAR_DEFAULT     = "2023"

# Load borough boundaries and project to UTM
gdf = gpd.read_file("data/nybb_25a/nybb.shp").to_crs(epsg=EPSG_NYC_UTM)
borough_union = gdf.geometry.union_all()

# Fetch and clip complaints around a point
def get_crime_data_api(lat, lon, radius_miles=0.25, year=YEAR_DEFAULT):
    rd = radius_miles / 69.0
    lat_min, lat_max = lat - rd, lat + rd
    lon_min, lon_max = lon - rd, lon + rd
    all_data = []
    offset = 0
    while True:
        params = {
            "$limit": API_LIMIT,
            "$offset": offset,
            "$select": "law_cat_cd, latitude, longitude",
            "$where": (
                f"cmplnt_fr_dt BETWEEN '{year}-01-01T00:00:00' "
                f"AND '{year}-12-31T23:59:59' AND "
                f"latitude BETWEEN {lat_min} AND {lat_max} AND longitude BETWEEN {lon_min} AND {lon_max}"
            )
        }
        r = requests.get(SODA_ENDPOINT, params=params)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        all_data.extend(chunk)
        offset += API_LIMIT
    # create buffer in UTM for accurate intersection
    ten = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=EPSG_NYC_UTM)
    buffer = ten.buffer(radius_miles * 1609.34)[0]
    # convert raw to GeoDataFrame and clip
    df = gpd.GeoDataFrame(
        all_data,
        geometry=gpd.points_from_xy(
            [float(r['longitude']) for r in all_data],
            [float(r['latitude'])  for r in all_data]
        ),
        crs="EPSG:4326"
    ).to_crs(epsg=EPSG_NYC_UTM)
    clipped = df[df.geometry.within(buffer)]
    area_m2 = buffer.intersection(borough_union).area
    return {'data': clipped.to_dict('records'), 'area': area_m2}

# Compute severity score per 1000 m²
def score_crime_severity(crime_dict):
    crimes = crime_dict['data']
    area = crime_dict['area']
    total_weight = sum(
        SEVERITY_WEIGHTS.get(rec.get('law_cat_cd') or 'UNKNOWN', 0)
        for rec in crimes
    )
    return total_weight / area * 1000

# Get borough area
def get_borough_area(borough):
    cmap = {"MANHATTAN":1, "BRONX":2, "BROOKLYN":3, "QUEENS":4, "STATEN ISLAND":5}
    code = cmap.get(borough.upper())
    return gdf[gdf['BoroCode'] == code].geometry.area.sum() if code else None

# Sample grid points and filter by high complaint count
def sample_high_density_grid(
    borough,
    required=35,
    spacing=1000,
    radius_miles=0.25,
    year=YEAR_DEFAULT
):
    cmap = {"MANHATTAN":1, "BRONX":2, "BROOKLYN":3, "QUEENS":4, "STATEN ISLAND":5}
    code = cmap.get(borough.upper())
    if not code:
        raise ValueError(f"Unknown borough '{borough}'")
    # union polygon for sampling
    poly = gdf[gdf['BoroCode'] == code].geometry.union_all()
    minx, miny, maxx, maxy = poly.bounds
    # generate and shuffle grid points
    pts = [Point(x, y)
           for x in np.arange(minx, maxx + 1, spacing)
           for y in np.arange(miny, maxy + 1, spacing)
           if poly.contains(Point(x, y))]
    random.shuffle(pts)
    # collect severity scores for high-density points
    scores = []
    for p in pts:
        if len(scores) >= required:
            break
        # convert to lat/lon for API
        lon, lat = gpd.GeoSeries([p], crs=gdf.crs).to_crs(epsg=4326).iloc[0].coords[0]
        info = get_crime_data_api(lat, lon, radius_miles, year)
        if len(info['data']) < 100:
            continue
        scores.append(score_crime_severity(info))
    if len(scores) < required:
        print(f"⚠️ Only found {len(scores)} points; consider adjusting grid or threshold.")
    return np.array(scores)

if __name__ == '__main__':
    borough = 'BROOKLYN'
    scores = sample_high_density_grid(borough)
    mean_sev = scores.mean() if len(scores) else float('nan')
    std_sev = scores.std(ddof=0) if len(scores) else float('nan')

    # Specific address: 621 Morgan Avenue, Greenpoint (approx coords)
    lat_morgan, lon_morgan = 40.725, -73.94
    morgan_info = get_crime_data_api(lat_morgan, lon_morgan)
    morgan_score = score_crime_severity(morgan_info)

    print(f"Sampled {len(scores)} points with >=100 complaints each")
    print(f"Mean Severity: {mean_sev:.2f} per 1000 m²")
    print(f"Std Dev:        {std_sev:.2f} per 1000 m²")
    print(f"Borough Area:   {get_borough_area(borough):.2f} m²")

    print("\n--- Greenpoint Address ---")
    print(f"Location: 621 Morgan Ave, Greenpoint")
    print(f"Area: {morgan_info['area']:.2f} m²")
    print(f"Severity Score: {morgan_score:.2f} per 1000 m²")
