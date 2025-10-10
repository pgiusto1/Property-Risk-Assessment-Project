<<<<<<< HEAD
# Property-Risk-Assessment-Project
=======
# 🗽 NYC Property Risk Assessment

An AI-powered tool to evaluate property-level risk in New York City by combining FEMA flood zone data, NYPD crime statistics, PLUTO property records, and Hurricane Evacuation Zones. Built with Python, Streamlit, LangChain, and ChromaDB.

---

## 🚀 Features

- 📍 Address-level property risk scoring
- 🌊 Flood risk analysis using FEMA and NYC evacuation data
- 🚓 Crime density assessment from NYPD open data
- 🏚️ Property condition insights from NYC PLUTO
- 🧠 Natural language querying via LangChain (RAG)
- 🗺️ Geospatial visualization using GeoPandas + Streamlit

---

## 🏗️ Project Structure

```bash
nyc-property-risk-assessment/
├── app/            # Streamlit interface
├── agents/         # LangChain agents (flood, crime, PLUTO)
├── data/           # Cleaned or sample data files
├── notebooks/      # Jupyter notebooks for data exploration
├── pipeline/       # ETL and preprocessing scripts
├── retrieval/      # Vector store + embedding logic
└── tests/          # Unit tests
>>>>>>> fa14552 (Initial comit)
