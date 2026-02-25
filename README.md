# NYC Property Risk Assessment

An AI-powered tool to evaluate property-level risk in New York City by integrating
FEMA Flood Vulnerability Index data, NYPD crime statistics, and NYC PLUTO property
records. Built with a multimodal LLM-RAG pipeline using Mistral-7B, ChromaDB,
LangChain, and Streamlit.

---

## Architecture

```
Address Input
     │
     ├─── flood_agent.py  ──► FVI scores (storm surge, tidal, FSHRI)
     ├─── crime_agent.py  ──► NYPD severity score + borough baseline
     └─── pluto_agent.py  ──► MapPLUTO property attributes (BBL lookup)
                │
                ▼
     retrieval/vector_store.py
       ChromaDB + sentence-transformers
       (FVI tract docs with engineered features)
                │
           RAG Retrieve
                │
                ▼
     pipeline/rag_pipeline.py
       Mistral-7B-Instruct (HuggingFace)
       Structured risk assessment: 0–100 score + reasoning
                │
                ▼
          app/app.py  (Streamlit UI)
```

---

## Features

- **Address-level risk scoring** — flood, crime, and property sub-scores (0–100)
- **Multimodal RAG** — ChromaDB vector store of 2,000+ NYC census tract documents, retrieved semantically at query time to ground Mistral-7B reasoning
- **Flood risk** — FEMA FVI storm surge & tidal projections (present / 2050s / 2080s) merged with census tract geometries via GeoPandas
- **Crime density** — NYPD complaint severity score (felony/misdemeanor/violation) within a 0.25 mi radius, normalized against a borough-wide grid sample
- **Property data** — NYC MapPLUTO building class, year built, assessed value, zoning
- **EDA & feature engineering** — `notebooks/eda.ipynb` covers FVI distributions, choropleth maps, composite risk scoring, and crime severity analysis

---

## Project Structure

```
nyc-property-risk-assessment/
├── agents/
│   ├── flood_agent.py      # Geocode + FVI lookup (GeoPandas spatial join)
│   ├── crime_agent.py      # NYPD Socrata API + grid-sampled borough baseline
│   └── pluto_agent.py      # MapPLUTO BBL lookup via NYC Open Data
├── retrieval/
│   └── vector_store.py     # ChromaDB ingestion + semantic retrieval
├── pipeline/
│   └── rag_pipeline.py     # Retrieve → augment → Mistral-7B generate
├── notebooks/
│   └── eda.ipynb           # EDA + feature engineering on FVI / crime data
├── app/
│   └── app.py              # Streamlit UI
├── data/                   # (gitignored) shapefiles + FVI CSV
└── requirements.txt
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your HuggingFace API token:

```bash
export HUGGINGFACE_TOKEN=hf_...
```

Optionally set an NYC Open Data app token (raises rate limits):

```bash
export NYC_OPEN_DATA_TOKEN=...
```

---

## Usage

### Streamlit app

```bash
streamlit run app/app.py
```

### RAG pipeline (CLI)

```bash
python pipeline/rag_pipeline.py
```

### Build the vector store index (one-time)

```bash
python retrieval/vector_store.py
```

### Individual agents

```bash
python agents/flood_agent.py   # flood vulnerability for 621 Morgan Ave
python agents/crime_agent.py   # crime severity for Brooklyn
python agents/pluto_agent.py   # PLUTO record for a BBL
```

---

## Data Sources

| Dataset | Source |
|---|---|
| NYC Flood Vulnerability Index (FVI) | [NYC Open Data](https://data.cityofnewyork.us) |
| NYC Census Tracts 2020 (nyct2020) | [NYC Department of City Planning](https://www.nyc.gov/site/planning/data-maps/open-data.page) |
| NYC Borough Boundaries (nybb) | NYC Department of City Planning |
| NYPD Complaint Data Historic | [NYC Open Data — qgea-i56i](https://data.cityofnewyork.us/resource/qgea-i56i.json) |
| MapPLUTO | [NYC Open Data — 64uk-42ks](https://data.cityofnewyork.us/resource/64uk-42ks.json) |
| Mistral-7B-Instruct-v0.1 | [HuggingFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) |
