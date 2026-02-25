"""
app.py — Streamlit interface for the NYC Property Risk Assessment pipeline.

Run with:
    streamlit run app/app.py

API key resolution (in priority order):
  1. st.secrets["GROQ_API_KEY"]  — set in Streamlit Cloud dashboard
  2. GROQ_API_KEY environment variable — for local development
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Populate env var from Streamlit secrets if running on Streamlit Cloud
if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from agents.flood_agent import get_flood_vulnerability_from_address
from agents.crime_agent import get_crime_data_api, score_crime_severity, sample_high_density_grid
from agents.pluto_agent import get_property_data_from_address
from pipeline.rag_pipeline import generate_risk_assessment

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Property Risk Assessment",
    layout="wide",
)

st.markdown("""
<style>
    @font-face {
        font-family: 'Clother';
        src: local('Clother');
    }
    html, body, [class*="css"], [class*="st-"], button, input, select, textarea {
        font-family: 'Clother', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("NYC Property Risk Assessment")

# ---------------------------------------------------------------------------
# Sidebar — address input
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Property Address")
    address_no = st.text_input("House number", value="621")
    street_name = st.text_input("Street name", value="Morgan Avenue")
    borough = st.selectbox(
        "Borough",
        options=["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"],
    )
    BOROUGH_CODES = {
        "Manhattan": 1, "Bronx": 2, "Brooklyn": 3, "Queens": 4, "Staten Island": 5
    }
    BOROUGH_NAMES = {
        "Manhattan": "MANHATTAN", "Bronx": "BRONX", "Brooklyn": "BROOKLYN",
        "Queens": "QUEENS", "Staten Island": "STATEN ISLAND",
    }
    borough_code = BOROUGH_CODES[borough]
    borough_upper = BOROUGH_NAMES[borough]

    run = st.button("Assess Risk", type="primary", use_container_width=True)

    st.divider()
    st.caption("Set `GROQ_API_KEY` in your environment for LLM generation.")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if not run:
    st.info("Enter a NYC address in the sidebar and click **Assess Risk** to begin.")
    st.stop()

address_str = f"{address_no} {street_name}, {borough}"
st.subheader(address_str)

col_flood, col_crime, col_pluto = st.columns(3)

# ── Flood data ──────────────────────────────────────────────────────────────
with col_flood:
    with st.spinner("Fetching flood vulnerability…"):
        flood = get_flood_vulnerability_from_address(address_no, street_name, borough_code)

    if "error" in flood:
        st.error(f"Flood data: {flood['error']}")
        st.stop()

    st.markdown("### Flood Risk")
    fshri = flood.get("FSHRI")
    ss_2050 = flood.get("FVI_storm_surge_2050s")

    def risk_label(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "No flood zone"
        import math
        if math.isnan(f):
            return "No flood zone"
        if f >= 4:
            return f"**{f:.1f}** (High)"
        if f >= 3:
            return f"**{f:.1f}** (Moderate)"
        return f"**{f:.1f}** (Low)"

    st.write(f"**Census tract:** `{flood.get('GEOID', 'N/A')}`")
    st.write(f"**FSHRI** (socioeconomic): {risk_label(fshri)}")
    st.write(f"**Storm surge (present):** {risk_label(flood.get('FVI_storm_surge_present'))}")
    st.write(f"**Storm surge (2050s):** {risk_label(ss_2050)}")
    st.write(f"**Storm surge (2080s):** {risk_label(flood.get('FVI_storm_surge_2080s'))}")
    st.write(f"**Tidal (2020s):** {risk_label(flood.get('FVI_tidal_2020s'))}")
    st.write(f"**Tidal (2050s):** {risk_label(flood.get('FVI_tidal_2050s'))}")

# ── Crime data ───────────────────────────────────────────────────────────────
with col_crime:
    with st.spinner("Fetching crime statistics…"):
        try:
            # Geocode to get coordinates (reuse flood agent geocode)
            from agents.flood_agent import geocode_address_1b
            coords = geocode_address_1b(address_no, street_name, borough_code)
            if coords:
                lat, lon = coords
                crime_info = get_crime_data_api(lat, lon)
                severity = score_crime_severity(crime_info)

                # Borough baseline
                with st.spinner("Sampling borough baseline (may take ~60s)…"):
                    boro_scores = sample_high_density_grid(borough_upper, required=20)
                boro_mean = float(boro_scores.mean()) if len(boro_scores) else 2.0
                boro_std  = float(boro_scores.std())  if len(boro_scores) else 1.0

                crime = {
                    "severity_score": severity,
                    "mean": boro_mean,
                    "std": boro_std,
                    "area": crime_info["area"],
                }
            else:
                crime = {"severity_score": None, "mean": None, "std": None, "area": None}
        except Exception as e:
            crime = {"severity_score": None, "mean": None, "std": None, "area": None}
            st.warning(f"Crime data fetch failed: {e}")

    st.markdown("### Crime Risk")
    sev = crime.get("severity_score")
    mean = crime.get("mean")
    std = crime.get("std")
    if sev is not None and mean is not None and std is not None:
        zscore = (sev - mean) / std if std else 0
        crime_tier = "High" if zscore > 1 else ("Moderate" if zscore > -0.5 else "Low")
        st.write(f"**Severity score:** `{sev:.2f}` per 1,000 m²")
        st.write(f"**Borough mean:** `{mean:.2f}` ± `{std:.2f}`")
        st.write(f"**Z-score:** `{zscore:+.2f}`")
        st.write(f"**Relative risk:** {crime_tier}")
    else:
        st.write("Crime data unavailable.")

# ── PLUTO data ───────────────────────────────────────────────────────────────
with col_pluto:
    with st.spinner("Fetching property record (PLUTO)…"):
        pluto = get_property_data_from_address(address_no, street_name, borough_code)

    st.markdown("### Property Record")
    if "error" not in pluto:
        st.write(f"**Building class:** `{pluto.get('bldgclass', 'N/A')}`")
        st.write(f"**Year built:** `{pluto.get('yearbuilt', 'N/A')}`")
        st.write(f"**Floors:** `{pluto.get('numfloors', 'N/A')}`")
        st.write(f"**Land use:** `{pluto.get('landuse', 'N/A')}`")
        st.write(f"**Assessed value:** `${pluto.get('assesstot', 'N/A')}`")
        st.write(f"**Zoning:** `{pluto.get('zonedist1', 'N/A')}`")
    else:
        st.write("PLUTO record not found.")
        pluto = None

# ── RAG Generation ──────────────────────────────────────────────────────────
st.divider()
st.subheader("AI Risk Assessment (Llama 3.1 via RAG)")

with st.expander("How scores are calculated"):
    st.markdown("""
**Overall score** = 0.40 × Flood + 0.35 × Crime + 0.25 × Property

| Score | Formula |
|---|---|
| **Flood (0–100)** | Weighted composite of FVI storm surge (present 20%, 2050s 50%, 2080s 30%) and tidal values, combined with FSHRI socioeconomic vulnerability (weights: surge 40%, tidal 30%, FSHRI 30%). Scaled from the 1–5 FVI range to 0–100. |
| **Crime (0–100)** | Sigmoid function applied to the z-score of the local severity rate vs. the borough mean: `100 / (1 + e^(−1.5 × (z − 0.3)))`. A score above 50 means crime is above the borough average. |
| **Property (0–100)** | Building age (40%), building class risk tier (30%), and assessed value per lot sq ft (30%, lower value = higher risk). |

All scores are computed deterministically from raw data before the LLM generates its narrative.
""")

if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "Set `GROQ_API_KEY` in your environment to enable LLM generation. "
        "The data above is still available without it."
    )
else:
    with st.spinner("Retrieving context from ChromaDB + generating assessment…"):
        try:
            result = generate_risk_assessment(
                flood=flood,
                crime=crime,
                address=address_str,
                property_data=pluto,
            )
            scores = result["scores"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Risk", f"{scores['overall']} / 100")
            m2.metric("Flood Risk",   f"{scores['flood']} / 100")
            m3.metric("Crime Risk",   f"{scores['crime']} / 100")
            m4.metric("Property Risk", f"{scores['property']} / 100")

            st.markdown(result["narrative"])
        except Exception as e:
            import traceback
            st.error(f"LLM generation failed: {e}")
            st.code(traceback.format_exc())
