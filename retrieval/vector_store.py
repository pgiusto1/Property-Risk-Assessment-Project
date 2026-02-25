"""
vector_store.py

ChromaDB-backed vector store for NYC flood risk context documents.

Each census tract in the FVI dataset is converted into a natural-language
text document that describes its flood vulnerability profile. These documents
are embedded with sentence-transformers (all-MiniLM-L6-v2) and stored in a
persistent ChromaDB collection. At query time, the pipeline retrieves the
most semantically relevant tract documents to provide retrieved context
for the Mistral-7B LLM.

Feature engineering applied during ingestion:
  - Composite flood risk score (storm surge + tidal, 2020s–2080s)
  - Normalized FSHRI socioeconomic vulnerability tier
  - Risk-level annotations added to each document for richer retrieval
"""

import os
from typing import List, Optional

import chromadb
import pandas as pd
import geopandas as gpd
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
FVI_CSV = "data/New_York_City_s_Flood_Vulnerability_Index.csv"
TRACTS_SHP = "data/nyct2020_25a/nyct2020.shp"
CHROMA_DIR = "data/chroma_db"

RISK_LEVEL = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}


class FloodRiskVectorStore:
    """
    Manages a ChromaDB collection of NYC census tract flood risk documents.

    Usage (ingestion – run once):
        store = FloodRiskVectorStore()
        store.ingest_fvi_data()

    Usage (retrieval):
        docs = store.query_by_risk_profile(fshri=3.0, ss_2050=2.0)
    """

    def __init__(self, persist_dir: str = CHROMA_DIR):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="flood_risk_docs",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_fvi_data(
        self,
        fvi_csv_path: str = FVI_CSV,
        tracts_path: str = TRACTS_SHP,
    ) -> int:
        """
        Load FVI CSV + census tract geometries, apply feature engineering,
        and upsert each tract as a text document into ChromaDB.

        Feature engineering steps:
          1. Composite storm surge score: weighted average of present / 2050s / 2080s
          2. Composite tidal score: mean of available tidal FVI columns
          3. Combined flood index: 0.4 * storm_surge + 0.3 * tidal + 0.3 * FSHRI
          4. Risk tier: 1-5 → "very low" … "very high"
        """
        fvi = pd.read_csv(fvi_csv_path)
        tracts = gpd.read_file(tracts_path).to_crs("EPSG:4326")
        tracts["GEOID"] = tracts["GEOID"].astype(str)
        fvi["GEOID"] = fvi["GEOID"].astype(str)
        merged = tracts.merge(fvi, on="GEOID", how="inner")

        # --- Feature engineering ----------------------------------------
        def _f(col, row):
            v = row.get(col)
            return float(v) if pd.notna(v) and v != "" else None

        def _safe_mean(*vals):
            valid = [v for v in vals if v is not None]
            return sum(valid) / len(valid) if valid else None

        docs, ids, metadatas = [], [], []

        for _, row in merged.iterrows():
            geoid = row["GEOID"]
            centroid = row.geometry.centroid

            fshri = _f("FSHRI", row)
            ss_pres = _f("FVI_storm_surge_present", row)
            ss_2050 = _f("FVI_storm_surge_2050s", row)
            ss_2080 = _f("FVI_storm_surge_2080s", row)
            t_2020 = _f("FVI_tidal_2020s", row)
            t_2050 = _f("FVI_tidal_2050s", row)
            t_2080 = _f("FVI_tidal_2080s", row)

            # Engineered features
            composite_surge = _safe_mean(ss_pres, ss_2050, ss_2080)
            composite_tidal = _safe_mean(t_2020, t_2050, t_2080)
            components = [
                (composite_surge, 0.4),
                (composite_tidal, 0.3),
                (fshri, 0.3),
            ]
            valid_w = [(v, w) for v, w in components if v is not None]
            if valid_w:
                total_w = sum(w for _, w in valid_w)
                combined_flood = sum(v * w for v, w in valid_w) / total_w
            else:
                combined_flood = None

            text = self._build_doc_text(
                geoid, fshri, ss_pres, ss_2050, ss_2080,
                t_2020, t_2050, t_2080, combined_flood
            )

            meta = {
                "GEOID": geoid,
                "lat": round(centroid.y, 6),
                "lon": round(centroid.x, 6),
                "FSHRI": fshri or 0.0,
                "FVI_storm_surge_2050s": ss_2050 or 0.0,
                "FVI_storm_surge_2080s": ss_2080 or 0.0,
                "combined_flood_index": combined_flood or 0.0,
            }

            docs.append(text)
            ids.append(geoid)
            metadatas.append(meta)

        BATCH = 200
        for i in range(0, len(docs), BATCH):
            self.collection.upsert(
                documents=docs[i : i + BATCH],
                ids=ids[i : i + BATCH],
                metadatas=metadatas[i : i + BATCH],
            )

        print(f"[VectorStore] Ingested {len(docs)} census tracts into ChromaDB.")
        return len(docs)

    @staticmethod
    def _build_doc_text(
        geoid, fshri, ss_pres, ss_2050, ss_2080,
        t_2020, t_2050, t_2080, combined_flood
    ) -> str:
        """
        Convert a census tract's FVI features into a natural-language document.
        Interpretive annotations are appended to improve semantic retrieval quality.
        """

        def fmt(v):
            return f"{v:.1f}" if v is not None else "N/A"

        def level(v):
            return RISK_LEVEL.get(round(v) if v else 0, "unknown")

        text = (
            f"NYC census tract {geoid} flood risk profile. "
            f"Socioeconomic vulnerability (FSHRI): {fmt(fshri)} / 5 — "
            f"{level(fshri)} community capacity to recover from floods. "
            f"Storm surge flood vulnerability: "
            f"present={fmt(ss_pres)}, 2050s={fmt(ss_2050)} ({level(ss_2050)}), "
            f"2080s={fmt(ss_2080)}. "
            f"Tidal flood vulnerability: "
            f"2020s={fmt(t_2020)}, 2050s={fmt(t_2050)}, 2080s={fmt(t_2080)}. "
        )

        if combined_flood is not None:
            text += (
                f"Combined flood risk index (engineered feature): "
                f"{combined_flood:.2f} / 5 — {level(combined_flood)} overall flood exposure. "
            )

        # Interpretive annotations for richer semantic search
        if ss_2050 is not None and ss_2050 >= 4:
            text += (
                "HIGH storm surge risk projected by 2050 due to sea level rise. "
                "Properties in this tract face elevated flood insurance costs and potential displacement risk. "
            )
        elif ss_2050 is not None and ss_2050 >= 3:
            text += (
                "MODERATE storm surge vulnerability by 2050. "
                "Flood insurance requirements are likely and basement flooding events may increase. "
            )
        elif ss_2050 is not None:
            text += "LOW storm surge risk currently projected through 2050s. "

        if fshri is not None and fshri >= 4:
            text += (
                "Highly vulnerable socioeconomic community with limited resources "
                "for flood recovery, insurance, or relocation. "
            )
        elif fshri is not None and fshri >= 3:
            text += "Moderate socioeconomic vulnerability; disaster recovery capacity may be constrained. "

        return text.strip()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, query_text: str, n_results: int = 3) -> List[str]:
        """Semantic search over all tract documents."""
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
        )
        return results["documents"][0] if results["documents"] else []

    def query_by_risk_profile(
        self,
        fshri: Optional[float],
        ss_2050: Optional[float],
        n_results: int = 3,
    ) -> List[str]:
        """
        Retrieve similar tract documents for a property's risk profile.
        The query is constructed from engineered risk labels so the semantic
        search matches contextually relevant tracts rather than just nearby ones.
        """
        import math
        def _safe_round(v):
            try:
                f = float(v)
                return round(f) if not math.isnan(f) else 0
            except (TypeError, ValueError):
                return 0

        ss_label = RISK_LEVEL.get(_safe_round(ss_2050), "unknown")
        fshri_label = RISK_LEVEL.get(_safe_round(fshri), "unknown")
        query = (
            f"NYC census tract with {fshri_label} socioeconomic vulnerability "
            f"and {ss_label} storm surge flood risk by 2050 — "
            f"FSHRI={fshri}, FVI_storm_surge_2050s={ss_2050}"
        )
        return self.query(query, n_results=n_results)


# ---------------------------------------------------------------------------
# CLI: ingest and smoke-test retrieval
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    store = FloodRiskVectorStore()

    if store.collection.count() == 0:
        print("Collection empty — ingesting FVI data...")
        store.ingest_fvi_data()
    else:
        print(f"Collection already has {store.collection.count()} documents.")

    print("\n--- Sample retrieval (FSHRI=3, storm surge 2050=2) ---")
    docs = store.query_by_risk_profile(fshri=3.0, ss_2050=2.0, n_results=3)
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] {doc[:400]}...")
