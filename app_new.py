# app.py
# Biomass Quantification Tool - Streamlit (Refactored to match Excel logic)
#
# Run:
#   pip install streamlit pandas
#   streamlit run app.py

from __future__ import annotations

import json
import geopandas as gpd
from shapely.geometry import mapping

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple, Dict, List
from streamlit_option_menu import option_menu
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

import pandas as pd
import streamlit as st


import base64
from pathlib import Path

def set_sidebar_background(image_path: str, overlay_opacity: float = 0.45) -> None:
    """
    Sets a background image for the Streamlit sidebar using CSS + base64 embedding.
    overlay_opacity: 0..1, higher = darker overlay for readability.
    """
    img_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Mime type (mimetypes often doesn't know avif reliably)
    suffix = Path(image_path).suffix.lower()
    if suffix == ".avif":
        mime = "image/avif"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/png"  # safe fallback

    st.markdown(
        f"""
        <style>
        /* Sidebar background image */
        [data-testid="stSidebar"] {{
            background-image:
              linear-gradient(rgba(0,0,0,{overlay_opacity}), rgba(0,0,0,{overlay_opacity})),
              url("data:{mime};base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Optional: make sidebar content sit nicely on top */
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}

        /* Optional: soften containers inside sidebar */
        [data-testid="stSidebar"] .stButton button,
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stSelectbox div,
        [data-testid="stSidebar"] .stCheckbox,
        [data-testid="stSidebar"] .stRadio {{
            background: rgba(255,255,255,0.08) !important;
            border-radius: 10px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _img_to_data_uri(image_path: str) -> str:
    img_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    suffix = Path(image_path).suffix.lower()
    if suffix == ".avif":
        mime = "image/avif"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/png"

    return f"data:{mime};base64,{b64}"


def set_sidebar_background(image_path: str, overlay_opacity: float = 0.45) -> None:
    uri = _img_to_data_uri(image_path)
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-image:
              linear-gradient(rgba(0,0,0,{overlay_opacity}), rgba(0,0,0,{overlay_opacity})),
              url("{uri}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_main_background(image_path: str, overlay_opacity: float = 0.25, fixed: bool = True) -> None:
    """
    Styles the main content area (right of the sidebar).
    overlay_opacity: 0..1 (higher = darker overlay for readability)
    fixed=True gives a nice parallax-style 'fixed' background.
    """
    uri = _img_to_data_uri(image_path)
    attachment = "fixed" if fixed else "scroll"

    st.markdown(
        f"""
        <style>
        /* Main screen background */
        [data-testid="stAppViewContainer"] {{
            background-image:
              linear-gradient(rgba(255,255,255,{overlay_opacity}), rgba(255,255,255,{overlay_opacity})),
              url("{uri}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: {attachment};
        }}

        /* Make the top header transparent so the background shows */
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        /* Optional: soften content blocks for readability */
        .stMarkdown, .stDataFrame, .stTable, .stMetric, .stAlert {{
            background: rgba(255,255,255,0.0);
        }}

        /* Optional: give common widgets a subtle glass card look */
        div[data-testid="stVerticalBlock"] > div:has(> .stDataFrame),
        div[data-testid="stVerticalBlock"] > div:has(> .stTable),
        div[data-testid="stVerticalBlock"] > div:has(> .stMetric),
        div[data-testid="stVerticalBlock"] > div:has(> .stAlert) {{
            background: rgba(255,255,255,0.72);
            border-radius: 14px;
            padding: 12px;
            backdrop-filter: blur(6px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Biomass Quantification App",
    #page_icon="🌿",
    layout="wide",
)

DB_PATH = "biomass_app.db"

FARMS_SHP_PATH = "data/Farm_boundaries.shp"  # change to your real path
FARMS_ID_FIELD = "DEEDS_ID"                # unique id field in shapefile (change)
FARMS_NAME_FIELD = "FARM_NAME"            # farm name/number field (change)
FARMS_OWNER_FIELD = "SURNAME"               # owner name field (change)


SITES = ["site_1", "site_2", "site_3"]  # canonical model sites

NAMIBIA_SOIL_TYPES = [
    "Arenosols (sandy)",
    "Calcisols",
    "Cambisols",
    "Fluvisols (alluvial)",
    "Gypsisols",
    "Leptosols (shallow/rocky)",
    "Luvisols",
    "Regosols",
    "Solonchaks (saline)",
    "Vertisols (clayey)",
    "Other / Unknown",
]

@st.cache_data(show_spinner=False)
def load_farms_gdf(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        # if your shapefile has no CRS, set it here appropriately
        # gdf = gdf.set_crs("EPSG:XXXX")
        pass
    # Ensure WGS84 for folium + lat/lon
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception:
        # if already 4326 or conversion fails, proceed
        pass
    # Ensure valid geometry
    gdf = gdf[gdf.geometry.notna()].copy()
    return gdf
def ensure_farms_schema() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(farms)")
    cols = {r[1] for r in cur.fetchall()}

    if "farm_feature_id" not in cols:
        cur.execute("ALTER TABLE farms ADD COLUMN farm_feature_id TEXT")
    if "farm_geom_geojson" not in cols:
        cur.execute("ALTER TABLE farms ADD COLUMN farm_geom_geojson TEXT")

    conn.commit()
    conn.close()

# -----------------------------
# Database helpers
# -----------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    # Users
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    # Projects
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_number TEXT UNIQUE NOT NULL,
            project_name TEXT,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (created_by) REFERENCES users(id)
        );
        """
    )

    # Farms
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS farms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER UNIQUE NOT NULL,
            farm_name_number TEXT NOT NULL,
            owner_name TEXT NOT NULL,
            title_deed_size_ha REAL,
            avg_rainfall_mm REAL,
            soil_type TEXT,
            farm_workers_avg INTEGER,
            latitude REAL,
            longitude REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        """
    )

    # Fieldwork headers
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fieldwork_headers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            phase TEXT NOT NULL CHECK(phase IN ('Pre-harvest', 'Post-harvest')),
            collector TEXT,
            collection_date TEXT,
            site_number TEXT,
            coord_south REAL,
            coord_east REAL,
            transect_length_m REAL DEFAULT 50,
            transect_width_m REAL DEFAULT 4,
            transect_area_m2 REAL DEFAULT 200,
            seedlings_lt_50cm INTEGER DEFAULT 0,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        """
    )

    # Fieldwork rows
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fieldwork_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            header_id INTEGER NOT NULL,
            hce REAL NOT NULL,
            height_class_label TEXT NOT NULL,
            smellifera INTEGER DEFAULT 0,
            vreficiens INTEGER DEFAULT 0,
            vluederitzii INTEGER DEFAULT 0,
            dcinerea INTEGER DEFAULT 0,
            cmopane INTEGER DEFAULT 0,
            tsericea INTEGER DEFAULT 0,
            other_acacias INTEGER DEFAULT 0,
            wood_kg REAL DEFAULT 0,
            leaf_kg REAL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (header_id) REFERENCES fieldwork_headers(id) ON DELETE CASCADE
        );
        """
    )

    # Project-level assumptions (single source of truth)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS project_assumptions (
            project_id INTEGER PRIMARY KEY,
            production_area_ha REAL DEFAULT 0,
            mai_rate REAL DEFAULT 0.0318,
            density_kg_m3 REAL DEFAULT 650,
            truck_load_t REAL DEFAULT 32,

            leaf_ratio REAL DEFAULT 0.15,
            te_height_m REAL DEFAULT 1.5,
            bushfeed_fraction REAL DEFAULT 0.15,

            biochar_yield REAL DEFAULT 0.3,
            charcoal_yield_drum REAL DEFAULT 0.2,
            charcoal_yield_retort REAL DEFAULT 0.3,

            slimf_m3_per_year REAL DEFAULT 5000,
            slimf_use_frac REAL DEFAULT 0.2,
            sustainable_frac REAL DEFAULT 0.65,

            wet_smellifera REAL DEFAULT 10.8,
            wet_vreficiens REAL DEFAULT 10.8,
            wet_dcinerea REAL DEFAULT 5.07,
            wet_cmopane REAL DEFAULT 4.74,
            wet_tsericea REAL DEFAULT 5.77,
            wet_other_acacia REAL DEFAULT 10.8,

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,

            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()

def migrate_db() -> None:
    """
    Safe, idempotent schema migration for existing SQLite DBs.
    Adds columns if they don't exist yet.
    """
    conn = get_conn()
    cur = conn.cursor()

    # ---- farms table ----
    cur.execute("PRAGMA table_info(farms)")
    farms_cols = {r[1] for r in cur.fetchall()}

    if "farm_feature_id" not in farms_cols:
        cur.execute("ALTER TABLE farms ADD COLUMN farm_feature_id TEXT")

    if "farm_geom_geojson" not in farms_cols:
        cur.execute("ALTER TABLE farms ADD COLUMN farm_geom_geojson TEXT")

    conn.commit()
    conn.close()


# -----------------------------
# Authentication (simple)
# -----------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username: str, password: str) -> Tuple[bool, str]:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
            (username.strip().lower(), hash_password(password), str(date.today())),
        )
        conn.commit()
        return True, "User created successfully."
    except sqlite3.IntegrityError:
        return False, "That username already exists."
    finally:
        conn.close()


def authenticate(username: str, password: str) -> Optional[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username.strip().lower(),),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    user_id, uname, pw_hash = row
    # if hash_password(password) != pw_hash:
    #     return None
    return {"id": user_id, "username": uname}

def list_fieldwork_points(project_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT id, phase, collection_date, collector, site_number,
               coord_south, coord_east, notes
        FROM fieldwork_headers
        WHERE project_id = ?
          AND coord_south IS NOT NULL AND coord_east IS NOT NULL
        ORDER BY id DESC;
        """,
        conn,
        params=(project_id,),
    )
    conn.close()
    return df


# -----------------------------
# Project/Farm CRUD
# -----------------------------
def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.write("")  # spacing


def list_projects(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT p.id, p.project_number, COALESCE(p.project_name,'') AS project_name, p.created_at
        FROM projects p
        WHERE p.created_by = ?
        ORDER BY p.created_at DESC, p.id DESC;
        """,
        conn,
        params=(user_id,),
    )
    conn.close()
    return df


def create_project(user_id: int, project_number: str, project_name: str) -> Tuple[bool, str]:
    project_number = project_number.strip()
    if not project_number:
        return False, "Project Number is required."
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO projects(project_number, project_name, created_by, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (project_number, project_name.strip(), user_id, str(date.today())),
        )
        project_id = cur.lastrowid

        # Create default assumptions row for this project (single source of truth)
        cur.execute(
            """
            INSERT OR IGNORE INTO project_assumptions(project_id, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (project_id, str(date.today()), str(date.today())),
        )

        conn.commit()
        return True, "Project created."
    except sqlite3.IntegrityError:
        return False, "That Project Number already exists."
    finally:
        conn.close()


def get_farm(project_id: int) -> Optional[dict]:
    conn = get_conn()
    cur = conn.cursor()

    # Detect current schema
    cur.execute("PRAGMA table_info(farms)")
    cols = {r[1] for r in cur.fetchall()}

    base_fields = [
        "farm_name_number",
        "owner_name",
        "title_deed_size_ha",
        "avg_rainfall_mm",
        "soil_type",
        "farm_workers_avg",
        "latitude",
        "longitude",
    ]

    optional_fields = []
    if "farm_feature_id" in cols:
        optional_fields.append("farm_feature_id")
    if "farm_geom_geojson" in cols:
        optional_fields.append("farm_geom_geojson")

    fields = base_fields + optional_fields

    sql = f"""
        SELECT {", ".join(fields)}
        FROM farms
        WHERE project_id = ?
    """
    cur.execute(sql, (project_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return dict(zip(fields, row))

def upsert_farm(project_id: int, farm: dict) -> None:
    conn = get_conn()
    cur = conn.cursor()

    # detect schema (so it's safe on older DBs)
    cur.execute("PRAGMA table_info(farms)")
    cols = {r[1] for r in cur.fetchall()}
    has_fid = "farm_feature_id" in cols
    has_geom = "farm_geom_geojson" in cols

    existing = get_farm(project_id)

    if existing is None:
        fields = [
            "project_id", "farm_name_number", "owner_name", "title_deed_size_ha", "avg_rainfall_mm",
            "soil_type", "farm_workers_avg", "latitude", "longitude"
        ]
        values = [
            project_id,
            farm["farm_name_number"],
            farm["owner_name"],
            farm["title_deed_size_ha"],
            farm["avg_rainfall_mm"],
            farm["soil_type"],
            farm["farm_workers_avg"],
            farm["latitude"],
            farm["longitude"],
        ]

        if has_fid:
            fields.append("farm_feature_id")
            values.append(farm.get("farm_feature_id"))
        if has_geom:
            fields.append("farm_geom_geojson")
            values.append(farm.get("farm_geom_geojson"))

        fields.append("created_at")
        values.append(str(date.today()))

        sql = f"INSERT INTO farms({', '.join(fields)}) VALUES ({', '.join(['?'] * len(fields))})"
        cur.execute(sql, values)

    else:
        set_fields = [
            "farm_name_number=?",
            "owner_name=?",
            "title_deed_size_ha=?",
            "avg_rainfall_mm=?",
            "soil_type=?",
            "farm_workers_avg=?",
            "latitude=?",
            "longitude=?",
        ]
        params = [
            farm["farm_name_number"],
            farm["owner_name"],
            farm["title_deed_size_ha"],
            farm["avg_rainfall_mm"],
            farm["soil_type"],
            farm["farm_workers_avg"],
            farm["latitude"],
            farm["longitude"],
        ]

        if has_fid:
            set_fields.append("farm_feature_id=?")
            params.append(farm.get("farm_feature_id"))
        if has_geom:
            set_fields.append("farm_geom_geojson=?")
            params.append(farm.get("farm_geom_geojson"))

        params.append(project_id)

        sql = f"UPDATE farms SET {', '.join(set_fields)} WHERE project_id=?"
        cur.execute(sql, params)

    conn.commit()
    conn.close()



# -----------------------------
# Assumptions CRUD
# -----------------------------
def get_assumptions(project_id: int) -> dict:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM project_assumptions WHERE project_id = ?", (project_id,))
    row = cur.fetchone()
    cols = [d[0] for d in cur.description]
    conn.close()
    if not row:
        # ensure exists
        conn = get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO project_assumptions(project_id, created_at, updated_at) VALUES (?, ?, ?)",
            (project_id, str(date.today()), str(date.today())),
        )
        conn.commit()
        conn.close()
        return get_assumptions(project_id)
    d = dict(zip(cols, row))
    return d


def update_assumptions(project_id: int, updates: dict) -> None:
    if not updates:
        return
    conn = get_conn()
    cur = conn.cursor()
    updates = dict(updates)
    updates["updated_at"] = str(date.today())

    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
    params = list(updates.values()) + [project_id]
    cur.execute(f"UPDATE project_assumptions SET {set_clause} WHERE project_id = ?", params)
    conn.commit()
    conn.close()


def assumptions_to_wet_map(a: dict) -> dict:
    # Map the app's captured species to the assumptions (wet kg/TE)
    # NOTE: V. luederitzii and "Other acacias" are treated as Other Acacia by default.
    return {
        "S. mellifera": float(a["wet_smellifera"]),
        "V. reficiens": float(a["wet_vreficiens"]),
        "D. cinerea": float(a["wet_dcinerea"]),
        "C. mopane": float(a["wet_cmopane"]),
        "T. sericea": float(a["wet_tsericea"]),
        "Other Acacia": float(a["wet_other_acacia"]),
        "V. luederitzii": float(a["wet_other_acacia"]),
        "Other acacias": float(a["wet_other_acacia"]),
    }


# -----------------------------
# Fieldwork defaults + logic (existing)
# -----------------------------
SPECIES_COLS = [
    "S. mellifera",
    "V. reficiens",
    "V. luederitzii",
    "D. cinerea",
    "C. mopane",
    "T. sericea",
    "Other acacias",
]

def default_hce_table() -> pd.DataFrame:
    rows = [
        (0.5, "Up to 0.5 m"),
        (1.0, "0.5–1 m"),
        (1.5, "1–1.5 m"),
        (2.0, "1.5–2 m"),
        (2.5, "2–2.5 m"),
        (3.0, "2.5–3 m"),
        (3.5, "3–3.5 m"),
        (4.0, "3.5–4 m"),
        (4.5, "4–4.5 m"),
        (5.0, "4.5–5 m"),
        (6.0, "5–6 m"),
        (7.0, "6–7 m"),
        (8.0, "7–8 m"),
        (9.0, "8–9 m"),
    ]
    df = pd.DataFrame(rows, columns=["HCE", "Height class"])
    for c in SPECIES_COLS:
        df[c] = 0
    df["Targeted Total HCE"] = 0.0
    df["Targeted Wood Estimates (kg/transect)"] = 0.0
    df["Targeted Leaf Estimates (kg/transect)"] = 0.0
    df["Total Targeted TE"] = 0.0
    return df


def compute_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    species_sum = df[SPECIES_COLS].sum(axis=1)
    df["Targeted Total HCE"] = (species_sum * df["HCE"]).round(3)
    df["Total Targeted TE"] = (
        df["Targeted Wood Estimates (kg/transect)"] + df["Targeted Leaf Estimates (kg/transect)"]
    ).round(3)
    return df


def save_fieldwork(project_id: int, phase: str, header: dict, table: pd.DataFrame) -> int:
    conn = get_conn()
    cur = conn.cursor()

    transect_area = (header["transect_length_m"] or 0) * (header["transect_width_m"] or 0)

    cur.execute(
        """
        INSERT INTO fieldwork_headers(
            project_id, phase, collector, collection_date, site_number,
            coord_south, coord_east, transect_length_m, transect_width_m, transect_area_m2,
            seedlings_lt_50cm, notes, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            phase,
            header.get("collector"),
            header.get("collection_date"),
            header.get("site_number"),
            header.get("coord_south"),
            header.get("coord_east"),
            header.get("transect_length_m", 50),
            header.get("transect_width_m", 4),
            transect_area,
            header.get("seedlings_lt_50cm", 0),
            header.get("notes"),
            str(date.today()),
        ),
    )
    header_id = cur.lastrowid

    for _, r in table.iterrows():
        cur.execute(
            """
            INSERT INTO fieldwork_rows(
                header_id, hce, height_class_label,
                smellifera, vreficiens, vluederitzii, dcinerea, cmopane, tsericea, other_acacias,
                wood_kg, leaf_kg, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                header_id,
                float(r["HCE"]),
                str(r["Height class"]),
                int(r["S. mellifera"]),
                int(r["V. reficiens"]),
                int(r["V. luederitzii"]),
                int(r["D. cinerea"]),
                int(r["C. mopane"]),
                int(r["T. sericea"]),
                int(r["Other acacias"]),
                float(r["Targeted Wood Estimates (kg/transect)"]),
                float(r["Targeted Leaf Estimates (kg/transect)"]),
                str(date.today()),
            ),
        )

    conn.commit()
    conn.close()
    return header_id


def list_fieldwork_headers(project_id: int, phase: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT id, collection_date, collector, site_number,
               coord_south, coord_east, transect_length_m, transect_width_m, transect_area_m2,
               seedlings_lt_50cm, notes, created_at
        FROM fieldwork_headers
        WHERE project_id = ? AND phase = ?
        ORDER BY id DESC;
        """,
        conn,
        params=(project_id, phase),
    )
    conn.close()
    return df


def load_fieldwork_table(header_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT hce AS "HCE",
               height_class_label AS "Height class",
               smellifera AS "S. mellifera",
               vreficiens AS "V. reficiens",
               vluederitzii AS "V. luederitzii",
               dcinerea AS "D. cinerea",
               cmopane AS "C. mopane",
               tsericea AS "T. sericea",
               other_acacias AS "Other acacias",
               wood_kg AS "Targeted Wood Estimates (kg/transect)",
               leaf_kg AS "Targeted Leaf Estimates (kg/transect)"
        FROM fieldwork_rows
        WHERE header_id = ?
        ORDER BY hce ASC;
        """,
        conn,
        params=(header_id,),
    )
    conn.close()

    df["Targeted Total HCE"] = 0.0
    df["Total Targeted TE"] = 0.0
    return compute_row_metrics(df)


# -----------------------------
# Pure compute functions (Excel model layer)
# -----------------------------
def average_nonzero(values: List[float]) -> float:
    nz = [v for v in values if v is not None and float(v) != 0.0]
    return float(sum(nz) / len(nz)) if nz else 0.0


def compute_te_biomass_density(te_counts: Dict[str, float], wet_kg_per_te: Dict[str, float], area_m2: float) -> float:
    """
    TE counts (per species) -> wet biomass kg (transect) -> biomass density kg/ha
    """
    total_kg = 0.0
    for sp, te in te_counts.items():
        total_kg += float(te) * float(wet_kg_per_te.get(sp, 0.0))
    if area_m2 <= 0:
        return 0.0
    kg_ha = (total_kg / float(area_m2)) * 10000.0
    return float(kg_ha)


def get_transect_te_counts(header_id: int) -> Dict[str, float]:
    """
    Sum TE counts per species across all height classes for a transect (header_id).
    """
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            SUM(smellifera) AS smellifera,
            SUM(vreficiens) AS vreficiens,
            SUM(vluederitzii) AS vluederitzii,
            SUM(dcinerea) AS dcinerea,
            SUM(cmopane) AS cmopane,
            SUM(tsericea) AS tsericea,
            SUM(other_acacias) AS other_acacias
        FROM fieldwork_rows
        WHERE header_id = ?
        """,
        conn,
        params=(header_id,),
    )
    conn.close()
    r = df.iloc[0].fillna(0)

    return {
        "S. mellifera": float(r["smellifera"]),
        "V. reficiens": float(r["vreficiens"]),
        "V. luederitzii": float(r["vluederitzii"]),
        "D. cinerea": float(r["dcinerea"]),
        "C. mopane": float(r["cmopane"]),
        "T. sericea": float(r["tsericea"]),
        "Other acacias": float(r["other_acacias"]),
    }


def list_transects_with_kg_ha(project_id: int, phase: str, assumptions: dict) -> pd.DataFrame:
    """
    Returns per-transect TE totals + wet biomass + kg/ha (foundational layer)
    """
    conn = get_conn()
    headers = pd.read_sql_query(
        """
        SELECT id, collection_date, collector, site_number, transect_area_m2
        FROM fieldwork_headers
        WHERE project_id = ? AND phase = ?
        ORDER BY id DESC
        """,
        conn,
        params=(project_id, phase),
    )
    conn.close()

    wet_map = assumptions_to_wet_map(assumptions)
    records = []
    for _, h in headers.iterrows():
        hid = int(h["id"])
        te_counts = get_transect_te_counts(hid)
        area_m2 = float(h["transect_area_m2"] or 0.0)
        kg_ha = compute_te_biomass_density(te_counts, wet_map, area_m2)
        total_te = sum(te_counts.values())
        records.append(
            {
                "header_id": hid,
                "collection_date": h["collection_date"],
                "collector": h["collector"],
                "site_number": str(h["site_number"] or ""),
                "transect_area_m2": area_m2,
                "total_te_all_species": float(total_te),
                "biomass_kg_ha": float(kg_ha),
            }
        )
    return pd.DataFrame(records)


def canonical_site(site_number_text: str) -> str:
    """
    Map user-entered site labels to site_1..site_3 where possible.
    If user uses "1", "Site 1", "site_1", etc., it will standardise.
    """
    s = (site_number_text or "").strip().lower().replace(" ", "").replace("-", "_")
    if s in {"1", "site1", "site_1"}:
        return "site_1"
    if s in {"2", "site2", "site_2"}:
        return "site_2"
    if s in {"3", "site3", "site_3"}:
        return "site_3"
    # fallback: keep raw but it won't be included in the model's standard 3 sites
    return s


def aggregate_site_biomass(project_id: int, assumptions: dict) -> Dict[str, Dict[str, float]]:
    """
    Site-level averages (mean across transects) for Pre and Post, ignoring zeros.
    Returns:
      { "site_1": {"pre_kg_ha":..., "post_kg_ha":...}, ... }
    """
    pre_df = list_transects_with_kg_ha(project_id, "Pre-harvest", assumptions)
    post_df = list_transects_with_kg_ha(project_id, "Post-harvest", assumptions)

    out = {s: {"pre_kg_ha": 0.0, "post_kg_ha": 0.0} for s in SITES}

    if not pre_df.empty:
        pre_df["site"] = pre_df["site_number"].apply(canonical_site)
    if not post_df.empty:
        post_df["site"] = post_df["site_number"].apply(canonical_site)

    for s in SITES:
        pre_vals = pre_df.loc[pre_df.get("site") == s, "biomass_kg_ha"].tolist() if not pre_df.empty else []
        post_vals = post_df.loc[post_df.get("site") == s, "biomass_kg_ha"].tolist() if not post_df.empty else []
        out[s]["pre_kg_ha"] = average_nonzero([float(v) for v in pre_vals]) if pre_vals else 0.0
        out[s]["post_kg_ha"] = average_nonzero([float(v) for v in post_vals]) if post_vals else 0.0

    return out


def compute_slimf_series(pre_kg_ha: float, area_ha: float, mai_rate: float, slimf_use_frac: float, years: int = 20) -> pd.DataFrame:
    """
    Implements the SLIMF 0..20-year simulation per your spec.
    """
    records = []
    B_prev = float(pre_kg_ha)

    # Year 0
    biomass_farm_t_y0 = (B_prev * area_ha) / 1000.0 if area_ha else 0.0
    records.append(
        {
            "year": 0,
            "B_kg_ha": B_prev,
            "mai_kg_ha": 0.0,
            "mai_farm_t": 0.0,
            "biomass_farm_t": biomass_farm_t_y0,
            "slimf_use_t": 0.0,
            "biomass_after_t": biomass_farm_t_y0,
            "mai_after_t": (biomass_farm_t_y0 * mai_rate) if area_ha else 0.0,
        }
    )

    for y in range(1, years + 1):
        mai_kg_ha = B_prev * mai_rate
        B = B_prev + mai_kg_ha

        mai_farm_t = (mai_kg_ha * area_ha) / 1000.0 if area_ha else 0.0
        biomass_farm_t = (B * area_ha) / 1000.0 if area_ha else 0.0

        slimf_use_t = mai_farm_t * slimf_use_frac
        biomass_after_t = biomass_farm_t - slimf_use_t
        mai_after_t = biomass_after_t * mai_rate

        records.append(
            {
                "year": y,
                "B_kg_ha": B,
                "mai_kg_ha": mai_kg_ha,
                "mai_farm_t": mai_farm_t,
                "biomass_farm_t": biomass_farm_t,
                "slimf_use_t": slimf_use_t,
                "biomass_after_t": biomass_after_t,
                "mai_after_t": mai_after_t,
            }
        )
        B_prev = B

    return pd.DataFrame(records)


def compute_endproducts(biomass_available_t: float, a: dict) -> dict:
    truck_load_t = float(a["truck_load_t"])
    bushfeed_fraction = float(a["bushfeed_fraction"])
    charcoal_yield_drum = float(a["charcoal_yield_drum"])
    charcoal_yield_retort = float(a["charcoal_yield_retort"])
    biochar_yield = float(a["biochar_yield"])

    bush_feed_t = biomass_available_t * bushfeed_fraction
    drum_charcoal_t = biomass_available_t * charcoal_yield_drum
    retort_charcoal_t = max(0.0, (biomass_available_t - bush_feed_t)) * charcoal_yield_retort
    biochar_t = biomass_available_t * biochar_yield

    return {
        "biomass_available_t": biomass_available_t,
        "bush_feed_t": bush_feed_t,
        "drum_charcoal_t": drum_charcoal_t,
        "retort_charcoal_t": retort_charcoal_t,
        "biochar_t": biochar_t,
        "drum_truckloads": (drum_charcoal_t / truck_load_t) if truck_load_t else 0.0,
        "retort_truckloads": (retort_charcoal_t / truck_load_t) if truck_load_t else 0.0,
    }


def compute_non_slimf_horizons(slimf_series: pd.DataFrame, area_ha: float, sustainable_frac: float, horizons=(5, 10, 15, 20)) -> pd.DataFrame:
    """
    Horizon tables using average biomass over 0..N (inclusive) from B_kg_ha series.
    """
    rows = []
    for N in horizons:
        sub = slimf_series.loc[(slimf_series["year"] >= 0) & (slimf_series["year"] <= N)].copy()
        avg_kg_ha = float(sub["B_kg_ha"].mean()) if not sub.empty else 0.0
        t_ha = avg_kg_ha / 1000.0
        farm_t = t_ha * area_ha
        harvestable_t_y = (farm_t / N) if N else 0.0
        sustainable_t_y = harvestable_t_y * sustainable_frac

        rows.append(
            {
                "horizon_years": N,
                "avg_B_kg_ha": avg_kg_ha,
                "avg_t_ha": t_ha,
                "avg_farm_t": farm_t,
                "allowable_biomass_t_per_year": harvestable_t_y,
                "sustainable_biomass_t_per_year": sustainable_t_y,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# UI Components
# -----------------------------
def login_screen() -> None:
    st.title("Biomass Quantification App") #🌿
    st.caption("Login to manage projects, farms, fieldwork, and biomass results (SLIMF / NON-SLIMF).")

    tab1, tab2 = st.tabs(["Login", "Create account"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            user = authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.success("Logged in.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_username = st.text_input("New username", key="new_username")
        new_password = st.text_input("New password", type="password", key="new_password")
        new_password2 = st.text_input("Confirm password", type="password", key="new_password2")
        if st.button("Create account"):
            if not new_username.strip():
                st.error("Username is required.")
            elif not new_password:
                st.error("Password is required.")
            elif new_password != new_password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(new_username, new_password)
                (st.success if ok else st.error)(msg)

def render_assumptions_editor_page(a: dict, project_id: int) -> dict:
    kp = f"assump_{project_id}_page"  # unique keys

    # st.subheader("Assumptions & Inputs")
    st.caption("These parameters drive all biomass calculations (TE → SLIMF → NON-SLIMF).")

    c1, c2 = st.columns(2)
    with c1:
        production_area_ha = st.number_input(
            "Production area (ha)",
            min_value=0.0,
            step=1.0,
            value=float(a.get("production_area_ha") or 0.0),
            key=f"{kp}_production_area_ha",
        )
    with c2:
        mai_rate = st.number_input(
            "MAI rate (biomass thickening)",
            min_value=0.0,
            step=0.0001,
            format="%.4f",
            value=float(a.get("mai_rate") or 0.0318),
            key=f"{kp}_mai_rate",
        )

    st.divider()

    st.markdown("### Density & operations")
    c3, c4, c5 = st.columns(3)

    with c3:
        density_presets = {
            "Logs / chunks (650 kg/m³)": 650.0,
            "Chips (450 kg/m³)": 450.0,
            "Dense hardwood (750 kg/m³)": 750.0,
            "Custom": float(a.get("density_kg_m3") or 650.0),
        }
        preset = st.selectbox(
            "Density preset",
            list(density_presets.keys()),
            index=0,
            key=f"{kp}_density_preset",
        )
        density_kg_m3 = density_presets[preset]
        if preset == "Custom":
            density_kg_m3 = st.number_input(
                "Custom density (kg/m³)",
                min_value=1.0,
                step=10.0,
                value=float(a.get("density_kg_m3") or 650.0),
                key=f"{kp}_density_custom",
            )

    with c4:
        truck_load_t = st.number_input(
            "Truck load (t)",
            min_value=1.0,
            step=1.0,
            value=float(a.get("truck_load_t") or 32.0),
            key=f"{kp}_truck_load_t",
        )

    with c5:
        slimf_m3_per_year = st.number_input(
            "SLIMF cap (m³/year)",
            min_value=0.0,
            step=100.0,
            value=float(a.get("slimf_m3_per_year") or 5000.0),
            key=f"{kp}_slimf_m3_per_year",
        )

    slimf_t_per_year = (slimf_m3_per_year * float(density_kg_m3)) / 1000.0
    st.info(f"SLIMF cap in tonnes/year (from density): **{slimf_t_per_year:,.2f} t/yr**")

    st.divider()

    st.markdown("### Yields & fractions")
    y1, y2, y3, y4 = st.columns(4)
    with y1:
        bushfeed_fraction = st.number_input(
            "Bush feed fraction",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(a.get("bushfeed_fraction") or 0.15),
            key=f"{kp}_bushfeed_fraction",
        )
    with y2:
        biochar_yield = st.number_input(
            "Biochar yield",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(a.get("biochar_yield") or 0.3),
            key=f"{kp}_biochar_yield",
        )
    with y3:
        charcoal_yield_drum = st.number_input(
            "Charcoal yield (drum)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(a.get("charcoal_yield_drum") or 0.2),
            key=f"{kp}_charcoal_yield_drum",
        )
    with y4:
        charcoal_yield_retort = st.number_input(
            "Charcoal yield (retort)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(a.get("charcoal_yield_retort") or 0.3),
            key=f"{kp}_charcoal_yield_retort",
        )

    st.divider()

    st.markdown("### Regulatory fractions")
    r1, r2 = st.columns(2)
    with r1:
        slimf_use_frac = st.number_input(
            "SLIMF allowable use (fraction of MAI)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(a.get("slimf_use_frac") or 0.2),
            key=f"{kp}_slimf_use_frac",
        )
    with r2:
        sustainable_frac = st.number_input(
            "Sustainable use (fraction)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(a.get("sustainable_frac") or 0.65),
            key=f"{kp}_sustainable_frac",
        )

    st.divider()

    st.markdown("### Wet kg per TE (species)")
    s1, s2, s3 = st.columns(3)
    with s1:
        wet_smellifera = st.number_input("S. mellifera (kg/TE)", min_value=0.0, step=0.1,
                                         value=float(a.get("wet_smellifera") or 10.8), key=f"{kp}_wet_smellifera")
        wet_vreficiens = st.number_input("V. reficiens (kg/TE)", min_value=0.0, step=0.1,
                                         value=float(a.get("wet_vreficiens") or 10.8), key=f"{kp}_wet_vreficiens")
    with s2:
        wet_dcinerea = st.number_input("D. cinerea (kg/TE)", min_value=0.0, step=0.1,
                                       value=float(a.get("wet_dcinerea") or 5.07), key=f"{kp}_wet_dcinerea")
        wet_cmopane = st.number_input("C. mopane (kg/TE)", min_value=0.0, step=0.1,
                                      value=float(a.get("wet_cmopane") or 4.74), key=f"{kp}_wet_cmopane")
    with s3:
        wet_tsericea = st.number_input("T. sericea (kg/TE)", min_value=0.0, step=0.1,
                                       value=float(a.get("wet_tsericea") or 5.77), key=f"{kp}_wet_tsericea")
        wet_other_acacia = st.number_input("Other Acacia (kg/TE)", min_value=0.0, step=0.1,
                                           value=float(a.get("wet_other_acacia") or 10.8), key=f"{kp}_wet_other_acacia")

    st.divider()

    if st.button("Save assumptions", type="primary", key=f"{kp}_save"): #💾 
        update_assumptions(
            project_id,
            {
                "production_area_ha": float(production_area_ha),
                "mai_rate": float(mai_rate),
                "density_kg_m3": float(density_kg_m3),
                "truck_load_t": float(truck_load_t),
                "bushfeed_fraction": float(bushfeed_fraction),
                "biochar_yield": float(biochar_yield),
                "charcoal_yield_drum": float(charcoal_yield_drum),
                "charcoal_yield_retort": float(charcoal_yield_retort),
                "slimf_use_frac": float(slimf_use_frac),
                "sustainable_frac": float(sustainable_frac),
                "slimf_m3_per_year": float(slimf_m3_per_year),
                "wet_smellifera": float(wet_smellifera),
                "wet_vreficiens": float(wet_vreficiens),
                "wet_dcinerea": float(wet_dcinerea),
                "wet_cmopane": float(wet_cmopane),
                "wet_tsericea": float(wet_tsericea),
                "wet_other_acacia": float(wet_other_acacia),
            },
        )
        st.success("Assumptions saved.")
        st.rerun()

    return get_assumptions(project_id)



# def sidebar_assumptions_editor(project_id: int) -> dict:
#     a = get_assumptions(project_id)
#     kp = f"assump_{project_id}_sidebar"
#     return render_assumptions_editor(st.sidebar, a, project_id, kp)



def sidebar_nav() -> str:
    with st.sidebar:
        # Header block (icon + username)
        user = st.session_state.get("user") or {"username": "unknown"}
        st.markdown("### Biomass Quantification")
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin-top:6px;margin-bottom:8px;">
              <div style="font-size:22px;">👤</div>
              <div style="line-height:1.1;">
                <div style="font-weight:700;">{user.get("username","")}</div>
                <div style="font-size:12px;opacity:0.75;">Signed in</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        selected = option_menu(
            menu_title=None,
            options=[
                "Projects",
                "Farm registration",
                "Map",
                "Assumptions & Inputs",
                "Fieldwork (Pre/Post)",
                "Results: SLIMF",
                "Results: NON-SLIMF",
                "Export",
                "Logout",
            ],
            icons=[
                "folder2-open",
                "house",
                "map",
                "sliders",
                "clipboard-data",
                "graph-up-arrow",
                "table",
                "download",
                "box-arrow-right",
            ],
            default_index=0,
            styles={
                "container": {"padding": "0.2rem 0.2rem", "border-radius": "12px"},
                "icon": {"font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px",
                    "padding": "10px 12px",
                    "border-radius": "10px",
                },
                "nav-link-selected": {"font-weight": "700"},
            },
        )

        st.divider()

        proj = st.session_state.get("active_project_number")
        if proj:
            st.success(f"Active project: **{proj}**")
        else:
            st.warning("No active project selected")

    return selected


def projects_page(user_id: int) -> None:
    page_header("Projects") #📁
    st.write("Create a new biomass quantification project or select an existing one.")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Create new project")
        project_number = st.text_input("Project Number", placeholder="e.g., BQ-2025-001")
        project_name = st.text_input("Project Name (optional)", placeholder="e.g., Erongo Farm Biomass Survey")
        if st.button("Create project", type="primary"):
            ok, msg = create_project(user_id, project_number, project_name)
            (st.success if ok else st.error)(msg)

    with right:
        st.subheader("Your projects")
        df = list_projects(user_id)
        if df.empty:
            st.info("No projects yet. Create one on the left.")
            return

        st.dataframe(df[["project_number", "project_name", "created_at"]], use_container_width=True)

        options = df["project_number"].tolist()
        chosen = st.selectbox("Select active project", options, index=0)
        if st.button("Set active project"):
            project_id = int(df.loc[df["project_number"] == chosen, "id"].iloc[0])
            st.session_state["active_project_id"] = project_id
            st.session_state["active_project_number"] = chosen
            st.success(f"Active project set to {chosen}.")


def farm_page() -> None:
    page_header("Farm registration")
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        st.warning("Please select an active project first (Projects page).")
        return

    st.write(f"Active project: **{st.session_state.get('active_project_number','')}**")

    existing = get_farm(project_id) or {}

    # ---- safe defaults (prevents UnboundLocalError) ----
    auto_name = ""
    auto_owner = ""
    auto_lat = None
    auto_lon = None
    auto_fid = ""
    picked_geom = None

    # ---- load shapefile + picker first ----
    gdf = None
    try:
        gdf = load_farms_gdf(FARMS_SHP_PATH)
    except Exception as e:
        st.warning(f"Farm shapefile not loaded. Check FARMS_SHP_PATH. ({e})")

    selected_feature_id = (existing.get("farm_feature_id") or "").strip()

    if gdf is not None and not gdf.empty and FARMS_ID_FIELD in gdf.columns:
        def make_label(row):
            nm = str(row.get(FARMS_NAME_FIELD, "")).strip()
            ow = str(row.get(FARMS_OWNER_FIELD, "")).strip()
            fid = str(row.get(FARMS_ID_FIELD, "")).strip()
            return f"{nm} | {ow} | ID={fid}"

        gdf = gdf.copy()
        gdf["_fid"] = gdf[FARMS_ID_FIELD].astype(str)
        gdf["_label"] = gdf.apply(make_label, axis=1)

        labels = gdf["_label"].tolist()
        default_idx = 0

        if selected_feature_id:
            match_idx = gdf.index[gdf["_fid"] == str(selected_feature_id)].tolist()
            if match_idx:
                default_idx = int(match_idx[0])

        picked = st.selectbox(
            "Select farm from Namibia farms layer",
            labels,
            index=default_idx,
            key=f"farm_pick_{project_id}",
        )

        picked_row = gdf.loc[gdf["_label"] == picked].iloc[0]
        picked_geom = picked_row.geometry
        picked_centroid = (
            picked_geom.representative_point()
            if hasattr(picked_geom, "representative_point")
            else picked_geom.centroid
        )

        auto_lat = float(picked_centroid.y)
        auto_lon = float(picked_centroid.x)
        auto_name = str(picked_row.get(FARMS_NAME_FIELD, "")).strip()
        auto_owner = str(picked_row.get(FARMS_OWNER_FIELD, "")).strip()
        auto_fid = str(picked_row.get(FARMS_ID_FIELD, "")).strip()

        st.info("Farm details will auto-fill from the boundary layer. You can still edit fields if needed.")
    else:
        st.caption("No farm boundary layer available for auto-fill (or missing ID field).")

    # ---- decide the defaults for inputs (existing > auto > blank) ----
    default_farm_name = existing.get("farm_name_number") or auto_name or ""
    default_owner = existing.get("owner_name") or auto_owner or ""

    lat_default = existing.get("latitude")
    lon_default = existing.get("longitude")

    default_lat = float(lat_default) if lat_default is not None else (float(auto_lat) if auto_lat is not None else 0.0)
    default_lon = float(lon_default) if lon_default is not None else (float(auto_lon) if auto_lon is not None else 0.0)

    c1, c2, c3 = st.columns(3)

    with c1:
        farm_name_number = st.text_input(
            "Farm Name and number",
            value=default_farm_name,
            key=f"farm_name_{project_id}",
        )
        owner_name = st.text_input(
            "Owner's Name",
            value=default_owner,
            key=f"farm_owner_{project_id}",
        )

        current_soil = existing.get("soil_type") or "Other / Unknown"
        soil_type = st.selectbox(
            "Soil Type",
            NAMIBIA_SOIL_TYPES,
            index=(
                NAMIBIA_SOIL_TYPES.index(current_soil)
                if current_soil in NAMIBIA_SOIL_TYPES
                else len(NAMIBIA_SOIL_TYPES) - 1
            ),
            key=f"farm_soil_{project_id}",
        )

    with c2:
        title_deed_size_ha = st.number_input(
            "Title Deed size (ha)",
            min_value=0.0,
            step=1.0,
            value=float(existing.get("title_deed_size_ha") or 0.0),
            key=f"farm_deed_{project_id}",
        )
        avg_rainfall_mm = st.number_input(
            "Average rainfall (mm)",
            min_value=0.0,
            step=10.0,
            value=float(existing.get("avg_rainfall_mm") or 0.0),
            key=f"farm_rain_{project_id}",
        )
        farm_workers_avg = st.number_input(
            "Farm Workers (Average)",
            min_value=0,
            step=1,
            value=int(existing.get("farm_workers_avg") or 0),
            key=f"farm_workers_{project_id}",
        )

    with c3:
        latitude = st.number_input(
            "Latitude",
            value=default_lat,
            format="%.6f",
            key=f"farm_lat_{project_id}",
        )
        longitude = st.number_input(
            "Longitude",
            value=default_lon,
            format="%.6f",
            key=f"farm_lon_{project_id}",
        )
        st.caption("Use Decimal Degrees. Example: -22.560000, 17.083611")

    if st.button("Save farm details", type="primary", key=f"save_farm_{project_id}"):
        if not farm_name_number.strip() or not owner_name.strip():
            st.error("Farm Name/number and Owner's Name are required.")
            return

        farm = {
            "farm_name_number": farm_name_number.strip(),
            "owner_name": owner_name.strip(),
            "title_deed_size_ha": float(title_deed_size_ha),
            "avg_rainfall_mm": float(avg_rainfall_mm),
            "soil_type": soil_type.strip(),
            "farm_workers_avg": int(farm_workers_avg),
            "latitude": float(latitude),
            "longitude": float(longitude),
            # use selected farm if available, else keep existing stored value
            "farm_feature_id": str(auto_fid or existing.get("farm_feature_id") or ""),
            "farm_geom_geojson": (
                json.dumps(mapping(picked_geom)) if picked_geom is not None else (existing.get("farm_geom_geojson") or None)
            ),
        }

        upsert_farm(project_id, farm)
        st.success("Farm registered/updated successfully.")


def map_page() -> None:
    st.header("Map (Voyager): Farms + Fieldwork")
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        st.warning("Please select an active project first (Projects page).")
        return

    # Load shapefile boundaries
    gdf = None
    try:
        gdf = load_farms_gdf(FARMS_SHP_PATH)
    except Exception as e:
        st.error(f"Could not load farms shapefile: {e}")

    # Load registered farm (for this project)
    reg = get_farm(project_id) or {}
    reg_fid = str(reg.get("farm_feature_id") or "").strip()

    # Load fieldwork points
    pts = list_fieldwork_points(project_id)
    if not pts.empty:
        pts = pts.copy()
        pts["lat"] = pd.to_numeric(pts["coord_south"], errors="coerce")
        pts["lon"] = pd.to_numeric(pts["coord_east"], errors="coerce")
        pts = pts.dropna(subset=["lat", "lon"])
    else:
        pts = pd.DataFrame(columns=["lat", "lon", "phase"])

    # Filters
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        show_pre = st.checkbox("Show Pre-harvest points", value=True, key=f"map_pre_{project_id}")
    with c2:
        show_post = st.checkbox("Show Post-harvest points", value=True, key=f"map_post_{project_id}")
    with c3:
        st.caption("Tip: farm boundaries are grey; your registered farm is highlighted.")

    phases = []
    if show_pre:
        phases.append("Pre-harvest")
    if show_post:
        phases.append("Post-harvest")
    pts = pts[pts["phase"].isin(phases)] if not pts.empty else pts

    # Decide map centre: registered farm centroid > points mean > Namibia default
    centre = [-22.57, 17.08]  # fallback (Windhoek-ish)
    if reg.get("latitude") is not None and reg.get("longitude") is not None and (reg.get("latitude") != 0 or reg.get("longitude") != 0):
        centre = [float(reg["latitude"]), float(reg["longitude"])]
    elif not pts.empty:
        centre = [float(pts["lat"].mean()), float(pts["lon"].mean())]
    elif gdf is not None and not gdf.empty:
        cc = gdf.unary_union.centroid
        centre = [float(cc.y), float(cc.x)]

    # Base map
    m = folium.Map(location=centre, zoom_start=6, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Voyager",
        max_zoom=20,
    ).add_to(m)

    # 1) All farm boundaries
    if gdf is not None and not gdf.empty:
        all_layer = folium.FeatureGroup(name="All farm boundaries", show=True)
        folium.GeoJson(
            data=gdf.__geo_interface__,
            name="All farms",
            style_function=lambda _: {
                "color": "#777777",
                "weight": 1,
                "fillOpacity": 0.05,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in [FARMS_NAME_FIELD, FARMS_OWNER_FIELD, FARMS_ID_FIELD] if c in gdf.columns],
                aliases=["Farm", "Owner", "ID"],
                sticky=False,
            ),
        ).add_to(all_layer)
        all_layer.add_to(m)

        # 2) Highlight registered farm (if found)
        if reg_fid and FARMS_ID_FIELD in gdf.columns:
            hit = gdf[gdf[FARMS_ID_FIELD].astype(str) == reg_fid]
            if not hit.empty:
                hi = folium.FeatureGroup(name="Registered farm", show=True)
                folium.GeoJson(
                    data=hit.__geo_interface__,
                    style_function=lambda _: {
                        "color": "#ff0000",
                        "weight": 3,
                        "fillOpacity": 0.12,
                    },
                ).add_to(hi)
                hi.add_to(m)

                # zoom to it a bit
                try:
                    b = hit.total_bounds  # [minx, miny, maxx, maxy]
                    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
                except Exception:
                    pass

    # 3) Fieldwork points
    cluster = MarkerCluster(name="Fieldwork points").add_to(m)

    def colour(phase: str) -> str:
        return "green" if phase == "Pre-harvest" else "blue"

    for _, r in pts.iterrows():
        popup_html = f"""
        <b>Record ID:</b> {r.get('id','')}<br>
        <b>Phase:</b> {r.get('phase','')}<br>
        <b>Date:</b> {r.get('collection_date') or ''}<br>
        <b>Collector:</b> {r.get('collector') or ''}<br>
        <b>Site:</b> {r.get('site_number') or ''}<br>
        <b>Lat/Lon:</b> {float(r['lat']):.6f}, {float(r['lon']):.6f}<br>
        <b>Notes:</b> {r.get('notes') or ''}
        """
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color=colour(str(r["phase"])),
            fill=True,
            fill_opacity=0.85,
            tooltip=f"{r.get('phase','')} | Site {r.get('site_number','')} | ID {r.get('id','')}",
            popup=folium.Popup(popup_html, max_width=350),
        ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, use_container_width=True, height=650)


def assumptions_inputs_page() -> None:
    st.header("Assumptions & Inputs") #⚙️ 

    project_id = st.session_state.get("active_project_id")
    if not project_id:
        st.warning("Please select an active project first.")
        return

    a = get_assumptions(project_id)
    render_assumptions_editor_page(a, project_id)



def fieldwork_page() -> None:
    page_header("Fieldwork capture (Pre- and Post-harvest)") #🧾 
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        st.warning("Please select an active project first (Projects page).")
        return

    st.write(f"Active project: **{st.session_state.get('active_project_number','')}**")

    phase = st.radio("Phase", ["Pre-harvest", "Post-harvest"], horizontal=True)

    st.subheader("1) Transect metadata")
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1])

    with m1:
        collector = st.text_input("Collector")
        collection_date = st.date_input("Date (D/M/Y)", value=date.today())
        site_number = st.text_input("Site number (use 1/2/3 or Site 1/2/3 for best results)")

    with m2:
        coord_south = st.number_input("Coordinates South (Decimal)", value=0.0, format="%.6f")
        coord_east = st.number_input("Coordinates East (Decimal)", value=0.0, format="%.6f")
        seedlings = st.number_input("Number of seedlings < 50 cm", min_value=0, step=1, value=0)

    with m3:
        transect_length = st.number_input("Transect length (m)", min_value=0.0, step=1.0, value=50.0)
        transect_width = st.number_input("Transect width (m)", min_value=0.0, step=1.0, value=4.0)
        transect_area = transect_length * transect_width
        st.metric("Transect area (m²)", f"{transect_area:.0f}")

    with m4:
        notes = st.text_area("Notes (optional)", height=110)

    st.subheader("2) Height Class Equivalent counts + estimates")
    st.caption(
        "Enter species counts per height class. Biomass densities (kg/ha) used in SLIMF/NON-SLIMF are derived from TE counts only."
    )

    if "fieldwork_df" not in st.session_state or st.session_state.get("fieldwork_phase") != phase:
        st.session_state["fieldwork_df"] = default_hce_table()
        st.session_state["fieldwork_phase"] = phase

    df = st.session_state["fieldwork_df"]

    editable_cols = ["HCE", "Height class"] + SPECIES_COLS + [
        "Targeted Wood Estimates (kg/transect)",
        "Targeted Leaf Estimates (kg/transect)",
    ]

    edited = st.data_editor(
        df[editable_cols],
        use_container_width=True,
        num_rows="fixed", 
        key=f"editor_{phase}",
    )

    computed = compute_row_metrics(edited)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Total Targeted HCE (sum)", f"{computed['Targeted Total HCE'].sum():.3f}")
    with s2:
        st.metric("Total Wood (kg/transect, sum)", f"{computed['Targeted Wood Estimates (kg/transect)'].sum():.3f}")
    with s3:
        st.metric("Total Leaf (kg/transect, sum)", f"{computed['Targeted Leaf Estimates (kg/transect)'].sum():.3f}")

    st.dataframe(
        computed[
            ["HCE", "Height class"] + SPECIES_COLS
            + ["Targeted Total HCE", "Targeted Wood Estimates (kg/transect)", "Targeted Leaf Estimates (kg/transect)", "Total Targeted TE"]
        ],
        use_container_width=True,
    )

    st.session_state["fieldwork_df"] = computed

    st.subheader("3) Save this transect")
    if st.button("Save transect", type="primary"):
        header = {
            "collector": collector.strip(),
            "collection_date": collection_date.strftime("%Y-%m-%d"),
            "site_number": site_number.strip(),
            "coord_south": float(coord_south),
            "coord_east": float(coord_east),
            "transect_length_m": float(transect_length),
            "transect_width_m": float(transect_width),
            "seedlings_lt_50cm": int(seedlings),
            "notes": notes.strip(),
        }

        if not header["site_number"]:
            st.error("Site number is required.")
            return

        header_id = save_fieldwork(project_id, phase, header, computed)
        st.success(f"Saved {phase} transect. Record ID: {header_id}")

    st.divider()

    st.subheader("4) View previously saved transects")
    saved_headers = list_fieldwork_headers(project_id, phase)
    if saved_headers.empty:
        st.info(f"No saved {phase} transects yet.")
        return

    st.dataframe(saved_headers, use_container_width=True)

    chosen_id = st.selectbox(
        "Load a transect record (by ID)",
        saved_headers["id"].tolist(),
        index=0,
    )

    if st.button("Load selected transect"):
        loaded = load_fieldwork_table(int(chosen_id))
        st.session_state["fieldwork_df"] = loaded
        st.success("Loaded into the table above. Scroll up to view/edit (editing does not overwrite saved data).")

    with st.expander("Export selected transect to CSV"):
        export_df = load_fieldwork_table(int(chosen_id))
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"fieldwork_{phase.lower().replace('-','_')}_header_{chosen_id}.csv",
            mime="text/csv",
        )


def slimf_results_page() -> None:
    page_header("Results: SLIMF (derived from Pre/Post TE counts)") #📈 
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        st.warning("Please select an active project first (Projects page).")
        return

    a = get_assumptions(project_id)
    area_ha = float(a["production_area_ha"] or 0.0)
    if area_ha <= 0:
        st.error("Set **Production area (ha)** in the sidebar under Assumptions & Inputs.")
        return

    site_aggs = aggregate_site_biomass(project_id, a)

    # Summary table: Pre/Post kg/ha + extracted
    summary_rows = []
    for s in SITES:
        pre = float(site_aggs[s]["pre_kg_ha"])
        post = float(site_aggs[s]["post_kg_ha"])
        extracted_kg_ha = pre - post
        extracted_t_farm = (extracted_kg_ha * area_ha) / 1000.0
        summary_rows.append(
            {
                "site": s,
                "pre_biomass_kg_ha": pre,
                "post_biomass_kg_ha": post,
                "extracted_kg_ha": extracted_kg_ha,
                "extracted_t_farm": extracted_t_farm,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    st.subheader("1) Site biomass densities (kg/ha) + extracted biomass")
    st.dataframe(summary_df, use_container_width=True)

    # Build per-site SLIMF series
    st.subheader("2) SLIMF series (0..20 years) per site")
    mai_rate = float(a["mai_rate"])
    slimf_use_frac = float(a["slimf_use_frac"])

    site_series = {}
    for s in SITES:
        pre_kg_ha = float(site_aggs[s]["pre_kg_ha"])
        if pre_kg_ha == 0:
            continue
        site_series[s] = compute_slimf_series(pre_kg_ha, area_ha, mai_rate, slimf_use_frac, years=20)

    if not site_series:
        st.warning("No SLIMF series can be computed yet — ensure you have Pre-harvest transects captured for Site 1/2/3.")
        return

    chosen_site = st.selectbox("View site", list(site_series.keys()), index=0)
    st.dataframe(site_series[chosen_site], use_container_width=True)

    # Biomass available under SLIMF conditions (use average annual slimf_use_t over years 1..20)
    st.subheader("3) Biomass available + end-products (site-averaged, Excel AVERAGEIF-style)")
    biomass_available_by_site = []
    for s, df in site_series.items():
        avg_allowable = float(df.loc[df["year"].between(1, 20), "slimf_use_t"].mean())
        biomass_available_by_site.append(avg_allowable)

    biomass_available_t = average_nonzero(biomass_available_by_site)
    endp = compute_endproducts(biomass_available_t, a)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Biomass available (t/yr)", f"{endp['biomass_available_t']:,.2f}")
    c2.metric("Bush feed (t/yr)", f"{endp['bush_feed_t']:,.2f}")
    c3.metric("Drum charcoal (t/yr)", f"{endp['drum_charcoal_t']:,.2f}")
    c4.metric("Retort charcoal (t/yr)", f"{endp['retort_charcoal_t']:,.2f}")

    c5, c6 = st.columns(2)
    c5.metric("Drum truckloads (per year)", f"{endp['drum_truckloads']:,.2f}")
    c6.metric("Retort truckloads (per year)", f"{endp['retort_truckloads']:,.2f}")

    with st.expander("Show end-product table"):
        st.dataframe(pd.DataFrame([endp]), use_container_width=True)

    # Optional: SLIMF cap conversion (m³→t)
    density_kg_m3 = float(a["density_kg_m3"] or 650.0)
    slimf_t_per_year = (float(a["slimf_m3_per_year"] or 5000.0) * density_kg_m3) / 1000.0
    st.caption(f"SLIMF cap (m³/year) converted to tonnes/year using density: **{slimf_t_per_year:,.2f} t/yr**")


def non_slimf_results_page() -> None:
    page_header("Results: NON-SLIMF (planning horizons)") #📊 
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        st.warning("Please select an active project first (Projects page).")
        return

    a = get_assumptions(project_id)
    area_ha = float(a["production_area_ha"] or 0.0)
    if area_ha <= 0:
        st.error("Set **Production area (ha)** in the sidebar under Assumptions & Inputs.")
        return

    site_aggs = aggregate_site_biomass(project_id, a)
    mai_rate = float(a["mai_rate"])
    slimf_use_frac = float(a["slimf_use_frac"])
    sustainable_frac = float(a["sustainable_frac"])

    # Build one representative series (site-averaged) by averaging non-zero Pre kg/ha first
    pre_vals = [float(site_aggs[s]["pre_kg_ha"]) for s in SITES]
    pre_avg_kg_ha = average_nonzero(pre_vals)
    if pre_avg_kg_ha == 0:
        st.warning("No NON-SLIMF planning possible yet — capture Pre-harvest transects for sites.")
        return

    rep_series = compute_slimf_series(pre_avg_kg_ha, area_ha, mai_rate, slimf_use_frac, years=20)

    st.subheader("1) Horizon table (5/10/15/20)")
    horizons_df = compute_non_slimf_horizons(rep_series, area_ha, sustainable_frac, horizons=(5, 10, 15, 20))
    st.dataframe(horizons_df, use_container_width=True)

    st.subheader("2) Charcoal ↔ biomass equivalence and sustainable (65%) summaries")
    # Workbook equivalence uses 0.2 frequently as biomass-to-charcoal ratio
    ratio = 0.2

    # Use allowable biomass per year from horizons and convert to charcoal equivalent
    rows = []
    for _, r in horizons_df.iterrows():
        N = int(r["horizon_years"])
        allowable_biomass_y = float(r["allowable_biomass_t_per_year"])
        allowable_charcoal_y = allowable_biomass_y * ratio
        sustainable_biomass_y = float(r["sustainable_biomass_t_per_year"])
        sustainable_charcoal_y = sustainable_biomass_y * ratio

        # also show inverse conversion to match Excel-style "/0.2"
        biomass_equiv_from_charcoal_allowable = allowable_charcoal_y / ratio if ratio else 0.0

        rows.append(
            {
                "horizon_years": N,
                "allowable_biomass_t_per_year": allowable_biomass_y,
                "allowable_charcoal_t_per_year (biomass*0.2)": allowable_charcoal_y,
                "sustainable_biomass_t_per_year (65%)": sustainable_biomass_y,
                "sustainable_charcoal_t_per_year (65%)": sustainable_charcoal_y,
                "check_biomass_equiv (charcoal/0.2)": biomass_equiv_from_charcoal_allowable,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("Show representative SLIMF series used for horizons (site-averaged Pre kg/ha)"):
        st.dataframe(rep_series, use_container_width=True)


def export_page() -> None:
    page_header("Export") #📤 
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        st.warning("Please select an active project first (Projects page).")
        return

    a = get_assumptions(project_id)

    st.subheader("1) Export TE-derived transect biomass densities")
    for phase in ["Pre-harvest", "Post-harvest"]:
        df = list_transects_with_kg_ha(project_id, phase, a)
        st.write(f"**{phase}**")
        if df.empty:
            st.info("No transects captured.")
            continue
        st.dataframe(df, use_container_width=True)
        st.download_button(
            f"Download {phase} transects (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"transects_{phase.lower().replace('-','_')}.csv",
            mime="text/csv",
        )

    st.subheader("2) Export SLIMF and NON-SLIMF summary tables (CSV)")
    area_ha = float(a["production_area_ha"] or 0.0)
    if area_ha <= 0:
        st.warning("Set Production area in assumptions to export SLIMF/NON-SLIMF.")
        return

    site_aggs = aggregate_site_biomass(project_id, a)
    pre_vals = [float(site_aggs[s]["pre_kg_ha"]) for s in SITES]
    pre_avg_kg_ha = average_nonzero(pre_vals)
    if pre_avg_kg_ha == 0:
        st.warning("No Pre kg/ha available yet.")
        return

    rep_series = compute_slimf_series(pre_avg_kg_ha, area_ha, float(a["mai_rate"]), float(a["slimf_use_frac"]), years=20)
    horizons_df = compute_non_slimf_horizons(rep_series, area_ha, float(a["sustainable_frac"]), horizons=(5, 10, 15, 20))

    st.download_button(
        "Download SLIMF series (CSV)",
        data=rep_series.to_csv(index=False).encode("utf-8"),
        file_name="slimf_series_site_avg.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download NON-SLIMF horizons (CSV)",
        data=horizons_df.to_csv(index=False).encode("utf-8"),
        file_name="non_slimf_horizons.csv",
        mime="text/csv",
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    init_db()
    migrate_db()

    if "user" not in st.session_state:
        st.session_state["user"] = None
        
    # DEV: force login (comment out later)
    # if "user" not in st.session_state or not st.session_state["user"]:
    #     st.session_state["user"] = {"id": 1, "username": "dev"}

    if not st.session_state["user"]:
        login_screen()
        return
        
    set_sidebar_background("assets/sidebar_bg.avif", overlay_opacity=0.45)
    set_main_background("assets/main_bg.avif", overlay_opacity=0.22, fixed=True)
    
    page = sidebar_nav()

    if page == "Logout":
        st.session_state.clear()
        st.success("Logged out.")
        st.rerun()

    user_id = int(st.session_state["user"]["id"])

    if page == "Projects":
        projects_page(user_id)
        return

    # # From here onward: require active project
    # project_id = st.session_state.get("active_project_id")
    # if not project_id:
    #     st.warning("Please select an active project first (Projects page).")
    #     return

    # # Sidebar assumptions editor is always available once a project is active
    # _ = sidebar_assumptions_editor(int(project_id))

    if page == "Farm registration":
        farm_page()
    elif page == "Map":
        map_page()
    elif page == "Assumptions & Inputs":
        assumptions_inputs_page()
    elif page == "Fieldwork (Pre/Post)":
        fieldwork_page()
    elif page == "Results: SLIMF":
        slimf_results_page()
    elif page == "Results: NON-SLIMF":
        non_slimf_results_page()
    elif page == "Export":
        export_page()


if __name__ == "__main__":
    main()





