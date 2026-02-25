"""
rag_pipeline.py

Multimodal LLM-RAG pipeline for NYC property risk assessment.

Architecture:
  1. Score    — compute deterministic sub-scores from flood/crime/PLUTO data
  2. Retrieve — query ChromaDB for flood risk context documents from
                semantically similar census tracts (via sentence-transformers)
  3. Augment  — build a structured prompt that passes pre-computed scores
                and retrieved context to the LLM
  4. Generate — Llama-3.1-8B (via Groq) writes narrative reasoning only;
                scores are fixed inputs, not LLM outputs

Scoring formulas
----------------
Flood (0-100):
  composite_surge = weighted avg of FVI_storm_surge_{present,2050s,2080s}
                    with weights (0.2, 0.5, 0.3) — 2050s most policy-relevant
  composite_tidal = mean of available FVI_tidal_{2020s,2050s,2080s}
  combined_fvi    = 0.4*surge + 0.3*tidal + 0.3*FSHRI  (each on 1–5 scale)
  flood_score     = (combined_fvi - 1) / 4 * 100

Crime (0-100):
  z = (severity_score - borough_mean) / borough_std
  crime_score = 100 / (1 + exp(-1.5 * (z - 0.3)))   ← sigmoid centred above 0

Property (0-100):
  age_score   = min(100, max(0, (age - 10) / 90 * 100))
  class_score = lookup table by first letter of bldgclass
  value_score = max(0, min(100, 100 - assessed_value_per_sqft))
  property_score = 0.4*age + 0.3*class + 0.3*value

Overall = 0.40*flood + 0.35*crime + 0.25*property
"""

import math
import os
import sys
import requests
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.vector_store import FloodRiskVectorStore

MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------------------------------------------------------------------------
# Vector store (lazy singleton)
# ---------------------------------------------------------------------------
_store: Optional[FloodRiskVectorStore] = None


def _get_store() -> FloodRiskVectorStore:
    global _store
    if _store is None:
        _store = FloodRiskVectorStore()
        if _store.collection.count() == 0:
            print("[RAG] ChromaDB collection empty — ingesting FVI data (one-time)...")
            _store.ingest_fvi_data()
    return _store


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------

def _fv(v) -> Optional[float]:
    """Return float or None, treating NaN as None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _flood_score(flood: Dict) -> int:
    """
    Compute flood risk score 0–100 from FVI data.
    Returns 50 (neutral) if no FVI data is present.
    """
    ss_pres = _fv(flood.get("FVI_storm_surge_present"))
    ss_2050 = _fv(flood.get("FVI_storm_surge_2050s"))
    ss_2080 = _fv(flood.get("FVI_storm_surge_2080s"))
    t_2020  = _fv(flood.get("FVI_tidal_2020s"))
    t_2050  = _fv(flood.get("FVI_tidal_2050s"))
    t_2080  = _fv(flood.get("FVI_tidal_2080s"))
    fshri   = _fv(flood.get("FSHRI"))

    # Composite storm surge: 2050s weighted most heavily
    surge_parts = [(ss_pres, 0.2), (ss_2050, 0.5), (ss_2080, 0.3)]
    surge_valid = [(v, w) for v, w in surge_parts if v is not None]
    composite_surge = (
        sum(v * w for v, w in surge_valid) / sum(w for _, w in surge_valid)
        if surge_valid else None
    )

    # Composite tidal: simple mean
    tidal_vals = [v for v in [t_2020, t_2050, t_2080] if v is not None]
    composite_tidal = sum(tidal_vals) / len(tidal_vals) if tidal_vals else None

    # Weighted combined flood index (1–5 scale)
    components = [(composite_surge, 0.4), (composite_tidal, 0.3), (fshri, 0.3)]
    valid = [(v, w) for v, w in components if v is not None]
    if not valid:
        return 50
    total_w = sum(w for _, w in valid)
    combined = sum(v * w for v, w in valid) / total_w

    return round(max(0, min(100, (combined - 1) / 4 * 100)))


# Building class first-letter risk lookup (0–100)
_BLDG_CLASS_RISK = {
    "A": 20, "B": 30, "C": 40, "D": 50,
    "E": 55, "F": 60, "G": 65, "H": 50,
    "I": 50, "J": 45, "K": 40, "L": 55,
    "M": 45, "O": 60, "Q": 50, "R": 25,
    "S": 40, "U": 70, "V": 70, "W": 55,
}


def _crime_score(crime: Dict) -> int:
    """
    Compute crime risk score 0–100 via sigmoid on z-score.
    z = (severity - borough_mean) / borough_std
    """
    sev  = _fv(crime.get("severity_score"))
    mean = _fv(crime.get("mean"))
    std  = _fv(crime.get("std"))

    if sev is None or mean is None or std is None:
        return 50
    z = (sev - mean) / std if std else 0.0
    score = 100 / (1 + math.exp(-1.5 * (z - 0.3)))
    return round(max(0, min(100, score)))


def _property_score(pluto: Optional[Dict]) -> int:
    """
    Compute property risk score 0–100 from PLUTO attributes.
    Factors: building age (40%), class risk (30%), assessed value density (30%).
    """
    if not pluto or "error" in pluto:
        return 50

    # Age component
    try:
        age = 2025 - int(float(pluto.get("yearbuilt", 1975)))
        age_score = max(0, min(100, (age - 10) / 90 * 100))
    except (TypeError, ValueError):
        age_score = 40.0

    # Building class component
    bc = str(pluto.get("bldgclass") or "").strip()
    class_score = float(_BLDG_CLASS_RISK.get(bc[0].upper() if bc else "", 40))

    # Assessed value per lot sq ft — lower value → higher risk
    try:
        val  = float(pluto.get("assesstot") or 0)
        area = float(pluto.get("lotarea") or 0)
        val_per_sqft = (val / area) if area > 0 else 0.0
        value_score = max(0, min(100, 100 - val_per_sqft))
    except (TypeError, ValueError):
        value_score = 50.0

    score = 0.40 * age_score + 0.30 * class_score + 0.30 * value_score
    return round(max(0, min(100, score)))


def compute_scores(
    flood: Dict,
    crime: Dict,
    property_data: Optional[Dict] = None,
) -> Dict:
    """
    Return deterministic risk scores as a dict:
      {"flood": int, "crime": int, "property": int, "overall": int}
    """
    f = _flood_score(flood)
    c = _crime_score(crime)
    p = _property_score(property_data)
    overall = round(0.40 * f + 0.35 * c + 0.25 * p)
    return {"flood": f, "crime": c, "property": p, "overall": overall}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    f = _fv(v)
    return "N/A" if f is None else f"{f:.2f}"


def _build_messages(
    address: str,
    flood: Dict,
    crime: Dict,
    property_data: Optional[Dict],
    context_docs: List[str],
    scores: Dict,
) -> List[Dict]:
    context_block = "\n\n".join(
        f"[Context {i + 1}] {doc}" for i, doc in enumerate(context_docs)
    )

    pluto_section = ""
    if property_data and "error" not in property_data:
        pluto_section = f"""
Property Record (NYC PLUTO):
  Building class : {property_data.get('bldgclass', 'N/A')}
  Year built     : {property_data.get('yearbuilt', 'N/A')}
  Floors         : {property_data.get('numfloors', 'N/A')}
  Land use       : {property_data.get('landuse', 'N/A')}
  Assessed value : ${property_data.get('assesstot', 'N/A')}
  Lot area (ft²) : {property_data.get('lotarea', 'N/A')}"""

    user_content = f"""You are a certified property risk analyst for New York City.

The following risk scores have been computed analytically from the raw data:

  Overall Risk Score : {scores['overall']} / 100
  Flood Risk         : {scores['flood']} / 100  (weight 40%)
  Crime Risk         : {scores['crime']} / 100  (weight 35%)
  Property Risk      : {scores['property']} / 100  (weight 25%)

Do NOT change these scores. Your task is to write a concise narrative explaining the reasoning behind each score, using the data and retrieved flood context below.

=== Retrieved Flood Context (similar NYC census tracts) ===
{context_block}

=== Target Property ===
Address: {address}

Flood Vulnerability Index (FVI):
  Census tract (GEOID)        : {flood.get('GEOID', 'N/A')}
  Storm surge — present       : {_fmt(flood.get('FVI_storm_surge_present'))}
  Storm surge — 2050s         : {_fmt(flood.get('FVI_storm_surge_2050s'))}  (scale 1–5)
  Storm surge — 2080s         : {_fmt(flood.get('FVI_storm_surge_2080s'))}
  Tidal flood — 2020s         : {_fmt(flood.get('FVI_tidal_2020s'))}
  Tidal flood — 2050s         : {_fmt(flood.get('FVI_tidal_2050s'))}
  Tidal flood — 2080s         : {_fmt(flood.get('FVI_tidal_2080s'))}
  Socioeconomic vulnerability : {_fmt(flood.get('FSHRI'))} / 5  (FSHRI)

Crime Statistics (0.25 mi radius):
  Severity score  : {_fmt(crime.get('severity_score'))} per 1,000 m²
  Borough mean    : {_fmt(crime.get('mean'))} per 1,000 m²
  Borough std dev : {_fmt(crime.get('std'))} per 1,000 m²
{pluto_section}

=== Instructions ===
Write exactly three sections. Keep each to 2–4 sentences.

FLOOD RISK ({scores['flood']}/100):
[Explain using FVI values and retrieved context. Note which components drove the score and any missing data.]

CRIME RISK ({scores['crime']}/100):
[Explain the severity z-score relative to the borough baseline. State whether the area is above or below average.]

PROPERTY RISK ({scores['property']}/100):
[Explain based on building age, class, and assessed value. Note any structural or market vulnerability factors.]"""

    return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# NaN sanitisation
# ---------------------------------------------------------------------------

def _sanitize(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        try:
            out[k] = None if (v is not None and math.isnan(float(v))) else v
        except (TypeError, ValueError):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_risk_assessment(
    flood: Dict,
    crime: Dict,
    address: str,
    property_data: Optional[Dict] = None,
) -> Dict:
    """
    Full RAG pipeline: score → retrieve → augment → generate.

    Returns
    -------
    dict with keys:
      "scores"    : {"flood", "crime", "property", "overall"}  — deterministic ints
      "narrative" : str — LLM-generated reasoning for each sub-score
    """
    flood = _sanitize(flood)
    crime = _sanitize(crime)

    scores = compute_scores(flood, crime, property_data)

    store = _get_store()
    context_docs = store.query_by_risk_profile(
        fshri=flood.get("FSHRI"),
        ss_2050=flood.get("FVI_storm_surge_2050s"),
        n_results=3,
    )

    messages = _build_messages(address, flood, crime, property_data, context_docs, scores)

    token = os.environ.get("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    narrative = r.json()["choices"][0]["message"]["content"]

    return {"scores": scores, "narrative": narrative}


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    address = "621 Morgan Avenue, Brooklyn"

    flood_data = {
        "GEOID": "36047033400",
        "FVI_storm_surge_present": None,
        "FVI_storm_surge_2050s": 2.0,
        "FVI_storm_surge_2080s": 2.0,
        "FVI_tidal_2020s": None,
        "FVI_tidal_2050s": None,
        "FVI_tidal_2080s": None,
        "FSHRI": 3.0,
    }

    crime_data = {
        "severity_score": 1.22,
        "mean": 2.00,
        "std": 1.15,
        "area": 507723.93,
    }

    result = generate_risk_assessment(flood_data, crime_data, address)
    print("\n--- Scores ---")
    for k, v in result["scores"].items():
        print(f"  {k}: {v}")
    print("\n--- Narrative ---")
    print(result["narrative"])
