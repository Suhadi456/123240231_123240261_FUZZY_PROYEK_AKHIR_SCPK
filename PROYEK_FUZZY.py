"""
==============================================================
  SPK Rekomendasi Hotel — Fuzzy Logic Mamdani
  Sistem Cerdas & Pendukung Keputusan (SCPK)
  Program Studi Informatika — UPN "Veteran" Yogyakarta
==============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPK Hotel — Fuzzy Mamdani",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS KUSTOM — TAMPILAN MODERN
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Card metric kustom */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 16px 20px;
    color: white !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}
div[data-testid="metric-container"] label {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.85rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
section[data-testid="stSidebar"] .stRadio > div > label {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 6px 12px;
    margin: 2px 0;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.18);
}
/* Tabel */
.dataframe thead th { background: #2c3e50 !important; color: white !important; }
/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
/* Tombol utama */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none; border-radius: 10px; color: white;
    font-size: 1.05rem; font-weight: 600;
    padding: 0.65rem 2rem;
    box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(102,126,234,0.55);
}
/* Info/success/warning box */
div[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data_baru.csv", on_bad_lines="skip", low_memory=False)
    cols = ["overall", "cleanliness", "value", "location", "rooms", "sleep_quality"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols).copy()
    df = df.reset_index(drop=True)
    df["hotel_id"] = df["hotel_id"].astype(str).str.split(".").str[0]
    return df

df = load_data()


# ─────────────────────────────────────────────────────────────
# SISTEM FUZZY MAMDANI
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def build_fuzzy_system():
    """
    Membangun sistem inferensi Fuzzy Mamdani dengan:
    - 5 variabel input (skala 1–5)
    - 1 variabel output skor (0–100)
    - Fungsi keanggotaan: rendah=trimf, sedang=trapmf, tinggi=trimf
    - Defuzzifikasi centroid
    - 243 rule base (3^5 kombinasi penuh)
    """
    # ── Variabel Input ──────────────────────────────────────
    cleanliness   = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "cleanliness")
    value         = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "value")
    location      = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "location")
    rooms         = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "rooms")
    sleep_quality = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "sleep_quality")

    # ── Fungsi Keanggotaan Input ────────────────────────────
    # rendah → trimf | sedang → trapmf | tinggi → trimf
    for var in [cleanliness, value, location, rooms, sleep_quality]:
        var["rendah"] = fuzz.trimf(var.universe,  [1.0, 1.0, 3.0])
        var["sedang"] = fuzz.trapmf(var.universe, [2.0, 2.5, 3.5, 4.0])
        var["tinggi"] = fuzz.trimf(var.universe,  [3.0, 5.0, 5.0])

    # ── Variabel Output ─────────────────────────────────────
    skor = ctrl.Consequent(np.arange(0, 101, 1), "skor")

    # Membership output: buruk=trimf, cukup=trapmf, bagus=trimf
    skor["buruk"] = fuzz.trimf(skor.universe,  [0,   0,  50])
    skor["cukup"] = fuzz.trapmf(skor.universe, [30, 40,  60, 70])
    skor["bagus"] = fuzz.trimf(skor.universe,  [50, 100, 100])

    # ══════════════════════════════════════════════════════════════
    # RULE BASE — 243 ATURAN EKSPLISIT (3^5 Kombinasi Penuh)
    # Singkatan: C=cleanliness, V=value, L=location, R=rooms, S=sleep_quality
    # Himpunan: T=tinggi, S=sedang, R=rendah
    # Output ditentukan berdasarkan SKOR DOMINANSI:
    #   Nilai T=2, S=1, R=0 → total 0–10
    #   total ≥ 7 → bagus | total 4–6 → cukup | total ≤ 3 → buruk
    # ══════════════════════════════════════════════════════════════
    rules = [
        # ────────────────────────────────────────────────────────
        # BLOK 1: cleanliness = TINGGI (81 rule, R1–R81)
        # ────────────────────────────────────────────────────────

        # cleanliness T | value T | location T (27 rule)
        # R1  C-T V-T L-T R-T S-T → total=10 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R2  C-T V-T L-T R-T S-S → total=9 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R3  C-T V-T L-T R-T S-R → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["bagus"]),
        # R4  C-T V-T L-T R-S S-T → total=9 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R5  C-T V-T L-T R-S S-S → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["bagus"]),
        # R6  C-T V-T L-T R-S S-R → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["bagus"]),
        # R7  C-T V-T L-T R-R S-T → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["bagus"]),
        # R8  C-T V-T L-T R-R S-S → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["bagus"]),
        # R9  C-T V-T L-T R-R S-R → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness T | value T | location S (9 rule)
        # R10 C-T V-T L-S R-T S-T → total=9 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R11 C-T V-T L-S R-T S-S → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R12 C-T V-T L-S R-T S-R → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["bagus"]),
        # R13 C-T V-T L-S R-S S-T → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R14 C-T V-T L-S R-S S-S → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["bagus"]),
        # R15 C-T V-T L-S R-S S-R → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R16 C-T V-T L-S R-R S-T → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["bagus"]),
        # R17 C-T V-T L-S R-R S-S → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R18 C-T V-T L-S R-R S-R → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness T | value T | location R (9 rule)
        # R19 C-T V-T L-R R-T S-T → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R20 C-T V-T L-R R-T S-S → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R21 C-T V-T L-R R-T S-R → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R22 C-T V-T L-R R-S S-T → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R23 C-T V-T L-R R-S S-S → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R24 C-T V-T L-R R-S S-R → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R25 C-T V-T L-R R-R S-T → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R26 C-T V-T L-R R-R S-S → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R27 C-T V-T L-R R-R S-R → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness T | value S | location T (9 rule)
        # R28 C-T V-S L-T R-T S-T → total=9 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R29 C-T V-S L-T R-T S-S → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R30 C-T V-S L-T R-T S-R → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["bagus"]),
        # R31 C-T V-S L-T R-S S-T → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R32 C-T V-S L-T R-S S-S → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["bagus"]),
        # R33 C-T V-S L-T R-S S-R → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R34 C-T V-S L-T R-R S-T → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["bagus"]),
        # R35 C-T V-S L-T R-R S-S → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R36 C-T V-S L-T R-R S-R → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness T | value S | location S (9 rule)
        # R37 C-T V-S L-S R-T S-T → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R38 C-T V-S L-S R-T S-S → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R39 C-T V-S L-S R-T S-R → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R40 C-T V-S L-S R-S S-T → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R41 C-T V-S L-S R-S S-S → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R42 C-T V-S L-S R-S S-R → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R43 C-T V-S L-S R-R S-T → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R44 C-T V-S L-S R-R S-S → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R45 C-T V-S L-S R-R S-R → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness T | value S | location R (9 rule)
        # R46 C-T V-S L-R R-T S-T → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R47 C-T V-S L-R R-T S-S → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R48 C-T V-S L-R R-T S-R → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R49 C-T V-S L-R R-S S-T → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R50 C-T V-S L-R R-S S-S → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R51 C-T V-S L-R R-S S-R → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R52 C-T V-S L-R R-R S-T → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R53 C-T V-S L-R R-R S-S → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R54 C-T V-S L-R R-R S-R → total=3 → buruk
        ctrl.Rule(cleanliness["tinggi"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness T | value R | location T (9 rule)
        # R55 C-T V-R L-T R-T S-T → total=8 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R56 C-T V-R L-T R-T S-S → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R57 C-T V-R L-T R-T S-R → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R58 C-T V-R L-T R-S S-T → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R59 C-T V-R L-T R-S S-S → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R60 C-T V-R L-T R-S S-R → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R61 C-T V-R L-T R-R S-T → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R62 C-T V-R L-T R-R S-S → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R63 C-T V-R L-T R-R S-R → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness T | value R | location S (9 rule)
        # R64 C-T V-R L-S R-T S-T → total=7 → bagus
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R65 C-T V-R L-S R-T S-S → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R66 C-T V-R L-S R-T S-R → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R67 C-T V-R L-S R-S S-T → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R68 C-T V-R L-S R-S S-S → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R69 C-T V-R L-S R-S S-R → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R70 C-T V-R L-S R-R S-T → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R71 C-T V-R L-S R-R S-S → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R72 C-T V-R L-S R-R S-R → total=3 → buruk
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness T | value R | location R (9 rule)
        # R73 C-T V-R L-R R-T S-T → total=6 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R74 C-T V-R L-R R-T S-S → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R75 C-T V-R L-R R-T S-R → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R76 C-T V-R L-R R-S S-T → total=5 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R77 C-T V-R L-R R-S S-S → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R78 C-T V-R L-R R-S S-R → total=3 → buruk
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R79 C-T V-R L-R R-R S-T → total=4 → cukup
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R80 C-T V-R L-R R-R S-S → total=3 → buruk
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R81 C-T V-R L-R R-R S-R → total=2 → buruk
        ctrl.Rule(cleanliness["tinggi"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # ────────────────────────────────────────────────────────
        # BLOK 2: cleanliness = SEDANG (81 rule, R82–R162)
        # ────────────────────────────────────────────────────────

        # cleanliness S | value T | location T (9 rule)
        # R82 C-S V-T L-T R-T S-T → total=9 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R83 C-S V-T L-T R-T S-S → total=8 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R84 C-S V-T L-T R-T S-R → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["bagus"]),
        # R85 C-S V-T L-T R-S S-T → total=8 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R86 C-S V-T L-T R-S S-S → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["bagus"]),
        # R87 C-S V-T L-T R-S S-R → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R88 C-S V-T L-T R-R S-T → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["bagus"]),
        # R89 C-S V-T L-T R-R S-S → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R90 C-S V-T L-T R-R S-R → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness S | value T | location S (9 rule)
        # R91 C-S V-T L-S R-T S-T → total=8 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R92 C-S V-T L-S R-T S-S → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R93 C-S V-T L-S R-T S-R → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R94 C-S V-T L-S R-S S-T → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R95 C-S V-T L-S R-S S-S → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R96 C-S V-T L-S R-S S-R → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R97 C-S V-T L-S R-R S-T → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R98 C-S V-T L-S R-R S-S → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R99 C-S V-T L-S R-R S-R → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness S | value T | location R (9 rule)
        # R100 C-S V-T L-R R-T S-T → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R101 C-S V-T L-R R-T S-S → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R102 C-S V-T L-R R-T S-R → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R103 C-S V-T L-R R-S S-T → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R104 C-S V-T L-R R-S S-S → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R105 C-S V-T L-R R-S S-R → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R106 C-S V-T L-R R-R S-T → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R107 C-S V-T L-R R-R S-S → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R108 C-S V-T L-R R-R S-R → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness S | value S | location T (9 rule)
        # R109 C-S V-S L-T R-T S-T → total=8 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R110 C-S V-S L-T R-T S-S → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R111 C-S V-S L-T R-T S-R → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R112 C-S V-S L-T R-S S-T → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R113 C-S V-S L-T R-S S-S → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R114 C-S V-S L-T R-S S-R → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R115 C-S V-S L-T R-R S-T → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R116 C-S V-S L-T R-R S-S → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R117 C-S V-S L-T R-R S-R → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness S | value S | location S (9 rule) — inti sedang
        # R118 C-S V-S L-S R-T S-T → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R119 C-S V-S L-S R-T S-S → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R120 C-S V-S L-S R-T S-R → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R121 C-S V-S L-S R-S S-T → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R122 C-S V-S L-S R-S S-S → total=5 → cukup (semua sedang)
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R123 C-S V-S L-S R-S S-R → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R124 C-S V-S L-S R-R S-T → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R125 C-S V-S L-S R-R S-S → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R126 C-S V-S L-S R-R S-R → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness S | value S | location R (9 rule)
        # R127 C-S V-S L-R R-T S-T → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R128 C-S V-S L-R R-T S-S → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R129 C-S V-S L-R R-T S-R → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R130 C-S V-S L-R R-S S-T → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R131 C-S V-S L-R R-S S-S → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R132 C-S V-S L-R R-S S-R → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R133 C-S V-S L-R R-R S-T → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R134 C-S V-S L-R R-R S-S → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R135 C-S V-S L-R R-R S-R → total=2 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness S | value R | location T (9 rule)
        # R136 C-S V-R L-T R-T S-T → total=7 → bagus
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R137 C-S V-R L-T R-T S-S → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R138 C-S V-R L-T R-T S-R → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R139 C-S V-R L-T R-S S-T → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R140 C-S V-R L-T R-S S-S → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R141 C-S V-R L-T R-S S-R → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R142 C-S V-R L-T R-R S-T → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R143 C-S V-R L-T R-R S-S → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R144 C-S V-R L-T R-R S-R → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness S | value R | location S (9 rule)
        # R145 C-S V-R L-S R-T S-T → total=6 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R146 C-S V-R L-S R-T S-S → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R147 C-S V-R L-S R-T S-R → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R148 C-S V-R L-S R-S S-T → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R149 C-S V-R L-S R-S S-S → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R150 C-S V-R L-S R-S S-R → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R151 C-S V-R L-S R-R S-T → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R152 C-S V-R L-S R-R S-S → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R153 C-S V-R L-S R-R S-R → total=2 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness S | value R | location R (9 rule)
        # R154 C-S V-R L-R R-T S-T → total=5 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R155 C-S V-R L-R R-T S-S → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R156 C-S V-R L-R R-T S-R → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["buruk"]),
        # R157 C-S V-R L-R R-S S-T → total=4 → cukup
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R158 C-S V-R L-R R-S S-S → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["buruk"]),
        # R159 C-S V-R L-R R-S S-R → total=2 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R160 C-S V-R L-R R-R S-T → total=3 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["buruk"]),
        # R161 C-S V-R L-R R-R S-S → total=2 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R162 C-S V-R L-R R-R S-R → total=1 → buruk
        ctrl.Rule(cleanliness["sedang"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # ────────────────────────────────────────────────────────
        # BLOK 3: cleanliness = RENDAH (81 rule, R163–R243)
        # ────────────────────────────────────────────────────────

        # cleanliness R | value T | location T (9 rule)
        # R163 C-R V-T L-T R-T S-T → total=8 → bagus
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R164 C-R V-T L-T R-T S-S → total=7 → bagus
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["bagus"]),
        # R165 C-R V-T L-T R-T S-R → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R166 C-R V-T L-T R-S S-T → total=7 → bagus
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["bagus"]),
        # R167 C-R V-T L-T R-S S-S → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R168 C-R V-T L-T R-S S-R → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R169 C-R V-T L-T R-R S-T → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R170 C-R V-T L-T R-R S-S → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R171 C-R V-T L-T R-R S-R → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["cukup"]),

        # cleanliness R | value T | location S (9 rule)
        # R172 C-R V-T L-S R-T S-T → total=7 → bagus
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R173 C-R V-T L-S R-T S-S → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R174 C-R V-T L-S R-T S-R → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R175 C-R V-T L-S R-S S-T → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R176 C-R V-T L-S R-S S-S → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R177 C-R V-T L-S R-S S-R → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R178 C-R V-T L-S R-R S-T → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R179 C-R V-T L-S R-R S-S → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R180 C-R V-T L-S R-R S-R → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness R | value T | location R (9 rule)
        # R181 C-R V-T L-R R-T S-T → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R182 C-R V-T L-R R-T S-S → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R183 C-R V-T L-R R-T S-R → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R184 C-R V-T L-R R-S S-T → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R185 C-R V-T L-R R-S S-S → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R186 C-R V-T L-R R-S S-R → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R187 C-R V-T L-R R-R S-T → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R188 C-R V-T L-R R-R S-S → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R189 C-R V-T L-R R-R S-R → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["tinggi"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness R | value S | location T (9 rule)
        # R190 C-R V-S L-T R-T S-T → total=7 → bagus
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["bagus"]),
        # R191 C-R V-S L-T R-T S-S → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R192 C-R V-S L-T R-T S-R → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R193 C-R V-S L-T R-S S-T → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R194 C-R V-S L-T R-S S-S → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R195 C-R V-S L-T R-S S-R → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["cukup"]),
        # R196 C-R V-S L-T R-R S-T → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R197 C-R V-S L-T R-R S-S → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["cukup"]),
        # R198 C-R V-S L-T R-R S-R → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness R | value S | location S (9 rule)
        # R199 C-R V-S L-S R-T S-T → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R200 C-R V-S L-S R-T S-S → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R201 C-R V-S L-S R-T S-R → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R202 C-R V-S L-S R-S S-T → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R203 C-R V-S L-S R-S S-S → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R204 C-R V-S L-S R-S S-R → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R205 C-R V-S L-S R-R S-T → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R206 C-R V-S L-S R-R S-S → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R207 C-R V-S L-S R-R S-R → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness R | value S | location R (9 rule)
        # R208 C-R V-S L-R R-T S-T → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R209 C-R V-S L-R R-T S-S → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R210 C-R V-S L-R R-T S-R → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["buruk"]),
        # R211 C-R V-S L-R R-S S-T → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R212 C-R V-S L-R R-S S-S → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["buruk"]),
        # R213 C-R V-S L-R R-S S-R → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R214 C-R V-S L-R R-R S-T → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["buruk"]),
        # R215 C-R V-S L-R R-R S-S → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R216 C-R V-S L-R R-R S-R → total=1 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["sedang"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness R | value R | location T (9 rule)
        # R217 C-R V-R L-T R-T S-T → total=6 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R218 C-R V-R L-T R-T S-S → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R219 C-R V-R L-T R-T S-R → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["cukup"]),
        # R220 C-R V-R L-T R-S S-T → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R221 C-R V-R L-T R-S S-S → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["cukup"]),
        # R222 C-R V-R L-T R-S S-R → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R223 C-R V-R L-T R-R S-T → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["cukup"]),
        # R224 C-R V-R L-T R-R S-S → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R225 C-R V-R L-T R-R S-R → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["tinggi"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness R | value R | location S (9 rule)
        # R226 C-R V-R L-S R-T S-T → total=5 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R227 C-R V-R L-S R-T S-S → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["cukup"]),
        # R228 C-R V-R L-S R-T S-R → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["buruk"]),
        # R229 C-R V-R L-S R-S S-T → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["cukup"]),
        # R230 C-R V-R L-S R-S S-S → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["buruk"]),
        # R231 C-R V-R L-S R-S S-R → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R232 C-R V-R L-S R-R S-T → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["buruk"]),
        # R233 C-R V-R L-S R-R S-S → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R234 C-R V-R L-S R-R S-R → total=1 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["sedang"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),

        # cleanliness R | value R | location R (9 rule) — semua rendah
        # R235 C-R V-R L-R R-T S-T → total=4 → cukup
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["tinggi"], skor["cukup"]),
        # R236 C-R V-R L-R R-T S-S → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["sedang"], skor["buruk"]),
        # R237 C-R V-R L-R R-T S-R → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["tinggi"] & sleep_quality["rendah"], skor["buruk"]),
        # R238 C-R V-R L-R R-S S-T → total=3 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["tinggi"], skor["buruk"]),
        # R239 C-R V-R L-R R-S S-S → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["sedang"], skor["buruk"]),
        # R240 C-R V-R L-R R-S S-R → total=1 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["sedang"] & sleep_quality["rendah"], skor["buruk"]),
        # R241 C-R V-R L-R R-R S-T → total=2 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["tinggi"], skor["buruk"]),
        # R242 C-R V-R L-R R-R S-S → total=1 → buruk
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["sedang"], skor["buruk"]),
        # R243 C-R V-R L-R R-R S-R → total=0 → buruk (semua rendah)
        ctrl.Rule(cleanliness["rendah"] & value["rendah"] & location["rendah"]
                  & rooms["rendah"] & sleep_quality["rendah"], skor["buruk"]),
    ]

    sistem_ctrl = ctrl.ControlSystem(rules)
    return sistem_ctrl, cleanliness, value, location, rooms, sleep_quality, skor


sistem_ctrl, fz_clean, fz_value, fz_loc, fz_rooms, fz_sleep, fz_skor = build_fuzzy_system()


# ─────────────────────────────────────────────────────────────
# FUNGSI HITUNG FUZZY (MAMDANI MURNI — TAHAP 1)
# ─────────────────────────────────────────────────────────────
def hitung_fuzzy(row):
    """
    Tahap 1: Inferensi Fuzzy Mamdani murni.
    Output: skor defuzzifikasi centroid (0–100), 1 desimal.
    Tidak dipengaruhi bobot — murni berdasarkan rule fuzzy.
    """
    try:
        sim = ctrl.ControlSystemSimulation(sistem_ctrl)
        sim.input["cleanliness"]   = float(np.clip(row["cleanliness"],   1, 5))
        sim.input["value"]         = float(np.clip(row["value"],         1, 5))
        sim.input["location"]      = float(np.clip(row["location"],      1, 5))
        sim.input["rooms"]         = float(np.clip(row["rooms"],         1, 5))
        sim.input["sleep_quality"] = float(np.clip(row["sleep_quality"], 1, 5))
        sim.compute()
        raw = sim.output["skor"]
        return round(float(np.clip(raw, 0, 100)), 1)
    except Exception:
        return np.nan


def normalisasi_bobot(w_c, w_v, w_l, w_r, w_s):
    """Normalisasi bobot agar total = 1.0."""
    total = w_c + w_v + w_l + w_r + w_s
    if total == 0:
        total = 1
    return w_c/total, w_v/total, w_l/total, w_r/total, w_s/total


def hitung_pref_score(row, n_c, n_v, n_l, n_r, n_s):
    """
    Tahap 2: Preference Score berdasarkan bobot pengguna.
    Rumus: weighted average nilai input × bobot → dipetakan ke skala 0–100.
    Input skala 1–5 → ((weighted_avg - 1) / 4) × 100
    """
    c = float(np.clip(row["cleanliness"],   1, 5))
    v = float(np.clip(row["value"],         1, 5))
    l = float(np.clip(row["location"],      1, 5))
    r = float(np.clip(row["rooms"],         1, 5))
    s = float(np.clip(row["sleep_quality"], 1, 5))
    w_avg = c*n_c + v*n_v + l*n_l + r*n_r + s*n_s
    return round(((w_avg - 1) / 4) * 100, 1)


def hitung_final_score(fuzzy_score, pref_score, alpha=0.7):
    """
    Tahap 3: Penggabungan Fuzzy + Bobot.
    final_score = alpha × fuzzy_score + (1-alpha) × pref_score
    Default alpha = 0.7 (fuzzy lebih dominan, bobot sebagai penyesuai).
    """
    if pd.isna(fuzzy_score) or pd.isna(pref_score):
        return np.nan
    return round(float(np.clip(alpha * fuzzy_score + (1 - alpha) * pref_score, 0, 100)), 1)


def label_rekomendasi(skor_val):
    """Konversi skor numerik ke label rekomendasi."""
    if pd.isna(skor_val):
        return "—"
    if skor_val >= 65:
        return "✅ Direkomendasikan"
    elif skor_val >= 40:
        return "⚠️ Cukup"
    else:
        return "❌ Tidak Direkomendasikan"


# ─────────────────────────────────────────────────────────────
# DAFTAR RULE EKSPLISIT — 243 ATURAN PENUH (3^5 kombinasi)
# Format: (antecedent_string, consequent_string)
# Kode: T=Tinggi, S=Sedang, R=Rendah
# ─────────────────────────────────────────────────────────────
RULES_LIST = [
    # ══ BLOK 1: Kebersihan = TINGGI (R1–R81) ══
    # C-T V-T L-T
    ("Kebersihan T & Nilai T & Lokasi T & Kamar T & Tidur T", "BAGUS"),   # R1
    ("Kebersihan T & Nilai T & Lokasi T & Kamar T & Tidur S", "BAGUS"),   # R2
    ("Kebersihan T & Nilai T & Lokasi T & Kamar T & Tidur R", "BAGUS"),   # R3
    ("Kebersihan T & Nilai T & Lokasi T & Kamar S & Tidur T", "BAGUS"),   # R4
    ("Kebersihan T & Nilai T & Lokasi T & Kamar S & Tidur S", "BAGUS"),   # R5
    ("Kebersihan T & Nilai T & Lokasi T & Kamar S & Tidur R", "BAGUS"),   # R6
    ("Kebersihan T & Nilai T & Lokasi T & Kamar R & Tidur T", "BAGUS"),   # R7
    ("Kebersihan T & Nilai T & Lokasi T & Kamar R & Tidur S", "BAGUS"),   # R8
    ("Kebersihan T & Nilai T & Lokasi T & Kamar R & Tidur R", "CUKUP"),   # R9
    # C-T V-T L-S
    ("Kebersihan T & Nilai T & Lokasi S & Kamar T & Tidur T", "BAGUS"),   # R10
    ("Kebersihan T & Nilai T & Lokasi S & Kamar T & Tidur S", "BAGUS"),   # R11
    ("Kebersihan T & Nilai T & Lokasi S & Kamar T & Tidur R", "BAGUS"),   # R12
    ("Kebersihan T & Nilai T & Lokasi S & Kamar S & Tidur T", "BAGUS"),   # R13
    ("Kebersihan T & Nilai T & Lokasi S & Kamar S & Tidur S", "BAGUS"),   # R14
    ("Kebersihan T & Nilai T & Lokasi S & Kamar S & Tidur R", "CUKUP"),   # R15
    ("Kebersihan T & Nilai T & Lokasi S & Kamar R & Tidur T", "BAGUS"),   # R16
    ("Kebersihan T & Nilai T & Lokasi S & Kamar R & Tidur S", "CUKUP"),   # R17
    ("Kebersihan T & Nilai T & Lokasi S & Kamar R & Tidur R", "CUKUP"),   # R18
    # C-T V-T L-R
    ("Kebersihan T & Nilai T & Lokasi R & Kamar T & Tidur T", "BAGUS"),   # R19
    ("Kebersihan T & Nilai T & Lokasi R & Kamar T & Tidur S", "BAGUS"),   # R20
    ("Kebersihan T & Nilai T & Lokasi R & Kamar T & Tidur R", "CUKUP"),   # R21
    ("Kebersihan T & Nilai T & Lokasi R & Kamar S & Tidur T", "BAGUS"),   # R22
    ("Kebersihan T & Nilai T & Lokasi R & Kamar S & Tidur S", "CUKUP"),   # R23
    ("Kebersihan T & Nilai T & Lokasi R & Kamar S & Tidur R", "CUKUP"),   # R24
    ("Kebersihan T & Nilai T & Lokasi R & Kamar R & Tidur T", "CUKUP"),   # R25
    ("Kebersihan T & Nilai T & Lokasi R & Kamar R & Tidur S", "CUKUP"),   # R26
    ("Kebersihan T & Nilai T & Lokasi R & Kamar R & Tidur R", "CUKUP"),   # R27
    # C-T V-S L-T
    ("Kebersihan T & Nilai S & Lokasi T & Kamar T & Tidur T", "BAGUS"),   # R28
    ("Kebersihan T & Nilai S & Lokasi T & Kamar T & Tidur S", "BAGUS"),   # R29
    ("Kebersihan T & Nilai S & Lokasi T & Kamar T & Tidur R", "BAGUS"),   # R30
    ("Kebersihan T & Nilai S & Lokasi T & Kamar S & Tidur T", "BAGUS"),   # R31
    ("Kebersihan T & Nilai S & Lokasi T & Kamar S & Tidur S", "BAGUS"),   # R32
    ("Kebersihan T & Nilai S & Lokasi T & Kamar S & Tidur R", "CUKUP"),   # R33
    ("Kebersihan T & Nilai S & Lokasi T & Kamar R & Tidur T", "BAGUS"),   # R34
    ("Kebersihan T & Nilai S & Lokasi T & Kamar R & Tidur S", "CUKUP"),   # R35
    ("Kebersihan T & Nilai S & Lokasi T & Kamar R & Tidur R", "CUKUP"),   # R36
    # C-T V-S L-S
    ("Kebersihan T & Nilai S & Lokasi S & Kamar T & Tidur T", "BAGUS"),   # R37
    ("Kebersihan T & Nilai S & Lokasi S & Kamar T & Tidur S", "BAGUS"),   # R38
    ("Kebersihan T & Nilai S & Lokasi S & Kamar T & Tidur R", "CUKUP"),   # R39
    ("Kebersihan T & Nilai S & Lokasi S & Kamar S & Tidur T", "BAGUS"),   # R40
    ("Kebersihan T & Nilai S & Lokasi S & Kamar S & Tidur S", "CUKUP"),   # R41
    ("Kebersihan T & Nilai S & Lokasi S & Kamar S & Tidur R", "CUKUP"),   # R42
    ("Kebersihan T & Nilai S & Lokasi S & Kamar R & Tidur T", "CUKUP"),   # R43
    ("Kebersihan T & Nilai S & Lokasi S & Kamar R & Tidur S", "CUKUP"),   # R44
    ("Kebersihan T & Nilai S & Lokasi S & Kamar R & Tidur R", "CUKUP"),   # R45
    # C-T V-S L-R
    ("Kebersihan T & Nilai S & Lokasi R & Kamar T & Tidur T", "BAGUS"),   # R46
    ("Kebersihan T & Nilai S & Lokasi R & Kamar T & Tidur S", "CUKUP"),   # R47
    ("Kebersihan T & Nilai S & Lokasi R & Kamar T & Tidur R", "CUKUP"),   # R48
    ("Kebersihan T & Nilai S & Lokasi R & Kamar S & Tidur T", "CUKUP"),   # R49
    ("Kebersihan T & Nilai S & Lokasi R & Kamar S & Tidur S", "CUKUP"),   # R50
    ("Kebersihan T & Nilai S & Lokasi R & Kamar S & Tidur R", "CUKUP"),   # R51
    ("Kebersihan T & Nilai S & Lokasi R & Kamar R & Tidur T", "CUKUP"),   # R52
    ("Kebersihan T & Nilai S & Lokasi R & Kamar R & Tidur S", "CUKUP"),   # R53
    ("Kebersihan T & Nilai S & Lokasi R & Kamar R & Tidur R", "BURUK"),   # R54
    # C-T V-R L-T
    ("Kebersihan T & Nilai R & Lokasi T & Kamar T & Tidur T", "BAGUS"),   # R55
    ("Kebersihan T & Nilai R & Lokasi T & Kamar T & Tidur S", "BAGUS"),   # R56
    ("Kebersihan T & Nilai R & Lokasi T & Kamar T & Tidur R", "CUKUP"),   # R57
    ("Kebersihan T & Nilai R & Lokasi T & Kamar S & Tidur T", "BAGUS"),   # R58
    ("Kebersihan T & Nilai R & Lokasi T & Kamar S & Tidur S", "CUKUP"),   # R59
    ("Kebersihan T & Nilai R & Lokasi T & Kamar S & Tidur R", "CUKUP"),   # R60
    ("Kebersihan T & Nilai R & Lokasi T & Kamar R & Tidur T", "CUKUP"),   # R61
    ("Kebersihan T & Nilai R & Lokasi T & Kamar R & Tidur S", "CUKUP"),   # R62
    ("Kebersihan T & Nilai R & Lokasi T & Kamar R & Tidur R", "CUKUP"),   # R63
    # C-T V-R L-S
    ("Kebersihan T & Nilai R & Lokasi S & Kamar T & Tidur T", "BAGUS"),   # R64
    ("Kebersihan T & Nilai R & Lokasi S & Kamar T & Tidur S", "CUKUP"),   # R65
    ("Kebersihan T & Nilai R & Lokasi S & Kamar T & Tidur R", "CUKUP"),   # R66
    ("Kebersihan T & Nilai R & Lokasi S & Kamar S & Tidur T", "CUKUP"),   # R67
    ("Kebersihan T & Nilai R & Lokasi S & Kamar S & Tidur S", "CUKUP"),   # R68
    ("Kebersihan T & Nilai R & Lokasi S & Kamar S & Tidur R", "CUKUP"),   # R69
    ("Kebersihan T & Nilai R & Lokasi S & Kamar R & Tidur T", "CUKUP"),   # R70
    ("Kebersihan T & Nilai R & Lokasi S & Kamar R & Tidur S", "CUKUP"),   # R71
    ("Kebersihan T & Nilai R & Lokasi S & Kamar R & Tidur R", "BURUK"),   # R72
    # C-T V-R L-R
    ("Kebersihan T & Nilai R & Lokasi R & Kamar T & Tidur T", "CUKUP"),   # R73
    ("Kebersihan T & Nilai R & Lokasi R & Kamar T & Tidur S", "CUKUP"),   # R74
    ("Kebersihan T & Nilai R & Lokasi R & Kamar T & Tidur R", "CUKUP"),   # R75
    ("Kebersihan T & Nilai R & Lokasi R & Kamar S & Tidur T", "CUKUP"),   # R76
    ("Kebersihan T & Nilai R & Lokasi R & Kamar S & Tidur S", "CUKUP"),   # R77
    ("Kebersihan T & Nilai R & Lokasi R & Kamar S & Tidur R", "BURUK"),   # R78
    ("Kebersihan T & Nilai R & Lokasi R & Kamar R & Tidur T", "CUKUP"),   # R79
    ("Kebersihan T & Nilai R & Lokasi R & Kamar R & Tidur S", "BURUK"),   # R80
    ("Kebersihan T & Nilai R & Lokasi R & Kamar R & Tidur R", "BURUK"),   # R81

    # ══ BLOK 2: Kebersihan = SEDANG (R82–R162) ══
    # C-S V-T L-T
    ("Kebersihan S & Nilai T & Lokasi T & Kamar T & Tidur T", "BAGUS"),   # R82
    ("Kebersihan S & Nilai T & Lokasi T & Kamar T & Tidur S", "BAGUS"),   # R83
    ("Kebersihan S & Nilai T & Lokasi T & Kamar T & Tidur R", "BAGUS"),   # R84
    ("Kebersihan S & Nilai T & Lokasi T & Kamar S & Tidur T", "BAGUS"),   # R85
    ("Kebersihan S & Nilai T & Lokasi T & Kamar S & Tidur S", "BAGUS"),   # R86
    ("Kebersihan S & Nilai T & Lokasi T & Kamar S & Tidur R", "CUKUP"),   # R87
    ("Kebersihan S & Nilai T & Lokasi T & Kamar R & Tidur T", "BAGUS"),   # R88
    ("Kebersihan S & Nilai T & Lokasi T & Kamar R & Tidur S", "CUKUP"),   # R89
    ("Kebersihan S & Nilai T & Lokasi T & Kamar R & Tidur R", "CUKUP"),   # R90
    # C-S V-T L-S
    ("Kebersihan S & Nilai T & Lokasi S & Kamar T & Tidur T", "BAGUS"),   # R91
    ("Kebersihan S & Nilai T & Lokasi S & Kamar T & Tidur S", "BAGUS"),   # R92
    ("Kebersihan S & Nilai T & Lokasi S & Kamar T & Tidur R", "CUKUP"),   # R93
    ("Kebersihan S & Nilai T & Lokasi S & Kamar S & Tidur T", "BAGUS"),   # R94
    ("Kebersihan S & Nilai T & Lokasi S & Kamar S & Tidur S", "CUKUP"),   # R95
    ("Kebersihan S & Nilai T & Lokasi S & Kamar S & Tidur R", "CUKUP"),   # R96
    ("Kebersihan S & Nilai T & Lokasi S & Kamar R & Tidur T", "CUKUP"),   # R97
    ("Kebersihan S & Nilai T & Lokasi S & Kamar R & Tidur S", "CUKUP"),   # R98
    ("Kebersihan S & Nilai T & Lokasi S & Kamar R & Tidur R", "CUKUP"),   # R99
    # C-S V-T L-R
    ("Kebersihan S & Nilai T & Lokasi R & Kamar T & Tidur T", "BAGUS"),   # R100
    ("Kebersihan S & Nilai T & Lokasi R & Kamar T & Tidur S", "CUKUP"),   # R101
    ("Kebersihan S & Nilai T & Lokasi R & Kamar T & Tidur R", "CUKUP"),   # R102
    ("Kebersihan S & Nilai T & Lokasi R & Kamar S & Tidur T", "CUKUP"),   # R103
    ("Kebersihan S & Nilai T & Lokasi R & Kamar S & Tidur S", "CUKUP"),   # R104
    ("Kebersihan S & Nilai T & Lokasi R & Kamar S & Tidur R", "CUKUP"),   # R105
    ("Kebersihan S & Nilai T & Lokasi R & Kamar R & Tidur T", "CUKUP"),   # R106
    ("Kebersihan S & Nilai T & Lokasi R & Kamar R & Tidur S", "CUKUP"),   # R107
    ("Kebersihan S & Nilai T & Lokasi R & Kamar R & Tidur R", "BURUK"),   # R108
    # C-S V-S L-T
    ("Kebersihan S & Nilai S & Lokasi T & Kamar T & Tidur T", "BAGUS"),   # R109
    ("Kebersihan S & Nilai S & Lokasi T & Kamar T & Tidur S", "BAGUS"),   # R110
    ("Kebersihan S & Nilai S & Lokasi T & Kamar T & Tidur R", "CUKUP"),   # R111
    ("Kebersihan S & Nilai S & Lokasi T & Kamar S & Tidur T", "BAGUS"),   # R112
    ("Kebersihan S & Nilai S & Lokasi T & Kamar S & Tidur S", "CUKUP"),   # R113
    ("Kebersihan S & Nilai S & Lokasi T & Kamar S & Tidur R", "CUKUP"),   # R114
    ("Kebersihan S & Nilai S & Lokasi T & Kamar R & Tidur T", "CUKUP"),   # R115
    ("Kebersihan S & Nilai S & Lokasi T & Kamar R & Tidur S", "CUKUP"),   # R116
    ("Kebersihan S & Nilai S & Lokasi T & Kamar R & Tidur R", "CUKUP"),   # R117
    # C-S V-S L-S
    ("Kebersihan S & Nilai S & Lokasi S & Kamar T & Tidur T", "BAGUS"),   # R118
    ("Kebersihan S & Nilai S & Lokasi S & Kamar T & Tidur S", "CUKUP"),   # R119
    ("Kebersihan S & Nilai S & Lokasi S & Kamar T & Tidur R", "CUKUP"),   # R120
    ("Kebersihan S & Nilai S & Lokasi S & Kamar S & Tidur T", "CUKUP"),   # R121
    ("Kebersihan S & Nilai S & Lokasi S & Kamar S & Tidur S", "CUKUP"),   # R122
    ("Kebersihan S & Nilai S & Lokasi S & Kamar S & Tidur R", "CUKUP"),   # R123
    ("Kebersihan S & Nilai S & Lokasi S & Kamar R & Tidur T", "CUKUP"),   # R124
    ("Kebersihan S & Nilai S & Lokasi S & Kamar R & Tidur S", "CUKUP"),   # R125
    ("Kebersihan S & Nilai S & Lokasi S & Kamar R & Tidur R", "BURUK"),   # R126
    # C-S V-S L-R
    ("Kebersihan S & Nilai S & Lokasi R & Kamar T & Tidur T", "CUKUP"),   # R127
    ("Kebersihan S & Nilai S & Lokasi R & Kamar T & Tidur S", "CUKUP"),   # R128
    ("Kebersihan S & Nilai S & Lokasi R & Kamar T & Tidur R", "CUKUP"),   # R129
    ("Kebersihan S & Nilai S & Lokasi R & Kamar S & Tidur T", "CUKUP"),   # R130
    ("Kebersihan S & Nilai S & Lokasi R & Kamar S & Tidur S", "CUKUP"),   # R131
    ("Kebersihan S & Nilai S & Lokasi R & Kamar S & Tidur R", "BURUK"),   # R132
    ("Kebersihan S & Nilai S & Lokasi R & Kamar R & Tidur T", "CUKUP"),   # R133
    ("Kebersihan S & Nilai S & Lokasi R & Kamar R & Tidur S", "BURUK"),   # R134
    ("Kebersihan S & Nilai S & Lokasi R & Kamar R & Tidur R", "BURUK"),   # R135
    # C-S V-R L-T
    ("Kebersihan S & Nilai R & Lokasi T & Kamar T & Tidur T", "BAGUS"),   # R136
    ("Kebersihan S & Nilai R & Lokasi T & Kamar T & Tidur S", "CUKUP"),   # R137
    ("Kebersihan S & Nilai R & Lokasi T & Kamar T & Tidur R", "CUKUP"),   # R138
    ("Kebersihan S & Nilai R & Lokasi T & Kamar S & Tidur T", "CUKUP"),   # R139
    ("Kebersihan S & Nilai R & Lokasi T & Kamar S & Tidur S", "CUKUP"),   # R140
    ("Kebersihan S & Nilai R & Lokasi T & Kamar S & Tidur R", "CUKUP"),   # R141
    ("Kebersihan S & Nilai R & Lokasi T & Kamar R & Tidur T", "CUKUP"),   # R142
    ("Kebersihan S & Nilai R & Lokasi T & Kamar R & Tidur S", "CUKUP"),   # R143
    ("Kebersihan S & Nilai R & Lokasi T & Kamar R & Tidur R", "BURUK"),   # R144
    # C-S V-R L-S
    ("Kebersihan S & Nilai R & Lokasi S & Kamar T & Tidur T", "CUKUP"),   # R145
    ("Kebersihan S & Nilai R & Lokasi S & Kamar T & Tidur S", "CUKUP"),   # R146
    ("Kebersihan S & Nilai R & Lokasi S & Kamar T & Tidur R", "CUKUP"),   # R147
    ("Kebersihan S & Nilai R & Lokasi S & Kamar S & Tidur T", "CUKUP"),   # R148
    ("Kebersihan S & Nilai R & Lokasi S & Kamar S & Tidur S", "CUKUP"),   # R149
    ("Kebersihan S & Nilai R & Lokasi S & Kamar S & Tidur R", "BURUK"),   # R150
    ("Kebersihan S & Nilai R & Lokasi S & Kamar R & Tidur T", "CUKUP"),   # R151
    ("Kebersihan S & Nilai R & Lokasi S & Kamar R & Tidur S", "BURUK"),   # R152
    ("Kebersihan S & Nilai R & Lokasi S & Kamar R & Tidur R", "BURUK"),   # R153
    # C-S V-R L-R
    ("Kebersihan S & Nilai R & Lokasi R & Kamar T & Tidur T", "CUKUP"),   # R154
    ("Kebersihan S & Nilai R & Lokasi R & Kamar T & Tidur S", "CUKUP"),   # R155
    ("Kebersihan S & Nilai R & Lokasi R & Kamar T & Tidur R", "BURUK"),   # R156
    ("Kebersihan S & Nilai R & Lokasi R & Kamar S & Tidur T", "CUKUP"),   # R157
    ("Kebersihan S & Nilai R & Lokasi R & Kamar S & Tidur S", "BURUK"),   # R158
    ("Kebersihan S & Nilai R & Lokasi R & Kamar S & Tidur R", "BURUK"),   # R159
    ("Kebersihan S & Nilai R & Lokasi R & Kamar R & Tidur T", "BURUK"),   # R160
    ("Kebersihan S & Nilai R & Lokasi R & Kamar R & Tidur S", "BURUK"),   # R161
    ("Kebersihan S & Nilai R & Lokasi R & Kamar R & Tidur R", "BURUK"),   # R162

    # ══ BLOK 3: Kebersihan = RENDAH (R163–R243) ══
    # C-R V-T L-T
    ("Kebersihan R & Nilai T & Lokasi T & Kamar T & Tidur T", "BAGUS"),   # R163
    ("Kebersihan R & Nilai T & Lokasi T & Kamar T & Tidur S", "BAGUS"),   # R164
    ("Kebersihan R & Nilai T & Lokasi T & Kamar T & Tidur R", "CUKUP"),   # R165
    ("Kebersihan R & Nilai T & Lokasi T & Kamar S & Tidur T", "BAGUS"),   # R166
    ("Kebersihan R & Nilai T & Lokasi T & Kamar S & Tidur S", "CUKUP"),   # R167
    ("Kebersihan R & Nilai T & Lokasi T & Kamar S & Tidur R", "CUKUP"),   # R168
    ("Kebersihan R & Nilai T & Lokasi T & Kamar R & Tidur T", "CUKUP"),   # R169
    ("Kebersihan R & Nilai T & Lokasi T & Kamar R & Tidur S", "CUKUP"),   # R170
    ("Kebersihan R & Nilai T & Lokasi T & Kamar R & Tidur R", "CUKUP"),   # R171
    # C-R V-T L-S
    ("Kebersihan R & Nilai T & Lokasi S & Kamar T & Tidur T", "BAGUS"),   # R172
    ("Kebersihan R & Nilai T & Lokasi S & Kamar T & Tidur S", "CUKUP"),   # R173
    ("Kebersihan R & Nilai T & Lokasi S & Kamar T & Tidur R", "CUKUP"),   # R174
    ("Kebersihan R & Nilai T & Lokasi S & Kamar S & Tidur T", "CUKUP"),   # R175
    ("Kebersihan R & Nilai T & Lokasi S & Kamar S & Tidur S", "CUKUP"),   # R176
    ("Kebersihan R & Nilai T & Lokasi S & Kamar S & Tidur R", "CUKUP"),   # R177
    ("Kebersihan R & Nilai T & Lokasi S & Kamar R & Tidur T", "CUKUP"),   # R178
    ("Kebersihan R & Nilai T & Lokasi S & Kamar R & Tidur S", "CUKUP"),   # R179
    ("Kebersihan R & Nilai T & Lokasi S & Kamar R & Tidur R", "BURUK"),   # R180
    # C-R V-T L-R
    ("Kebersihan R & Nilai T & Lokasi R & Kamar T & Tidur T", "CUKUP"),   # R181
    ("Kebersihan R & Nilai T & Lokasi R & Kamar T & Tidur S", "CUKUP"),   # R182
    ("Kebersihan R & Nilai T & Lokasi R & Kamar T & Tidur R", "CUKUP"),   # R183
    ("Kebersihan R & Nilai T & Lokasi R & Kamar S & Tidur T", "CUKUP"),   # R184
    ("Kebersihan R & Nilai T & Lokasi R & Kamar S & Tidur S", "CUKUP"),   # R185
    ("Kebersihan R & Nilai T & Lokasi R & Kamar S & Tidur R", "BURUK"),   # R186
    ("Kebersihan R & Nilai T & Lokasi R & Kamar R & Tidur T", "CUKUP"),   # R187
    ("Kebersihan R & Nilai T & Lokasi R & Kamar R & Tidur S", "BURUK"),   # R188
    ("Kebersihan R & Nilai T & Lokasi R & Kamar R & Tidur R", "BURUK"),   # R189
    # C-R V-S L-T
    ("Kebersihan R & Nilai S & Lokasi T & Kamar T & Tidur T", "BAGUS"),   # R190
    ("Kebersihan R & Nilai S & Lokasi T & Kamar T & Tidur S", "CUKUP"),   # R191
    ("Kebersihan R & Nilai S & Lokasi T & Kamar T & Tidur R", "CUKUP"),   # R192
    ("Kebersihan R & Nilai S & Lokasi T & Kamar S & Tidur T", "CUKUP"),   # R193
    ("Kebersihan R & Nilai S & Lokasi T & Kamar S & Tidur S", "CUKUP"),   # R194
    ("Kebersihan R & Nilai S & Lokasi T & Kamar S & Tidur R", "CUKUP"),   # R195
    ("Kebersihan R & Nilai S & Lokasi T & Kamar R & Tidur T", "CUKUP"),   # R196
    ("Kebersihan R & Nilai S & Lokasi T & Kamar R & Tidur S", "CUKUP"),   # R197
    ("Kebersihan R & Nilai S & Lokasi T & Kamar R & Tidur R", "BURUK"),   # R198
    # C-R V-S L-S
    ("Kebersihan R & Nilai S & Lokasi S & Kamar T & Tidur T", "CUKUP"),   # R199
    ("Kebersihan R & Nilai S & Lokasi S & Kamar T & Tidur S", "CUKUP"),   # R200
    ("Kebersihan R & Nilai S & Lokasi S & Kamar T & Tidur R", "CUKUP"),   # R201
    ("Kebersihan R & Nilai S & Lokasi S & Kamar S & Tidur T", "CUKUP"),   # R202
    ("Kebersihan R & Nilai S & Lokasi S & Kamar S & Tidur S", "CUKUP"),   # R203
    ("Kebersihan R & Nilai S & Lokasi S & Kamar S & Tidur R", "BURUK"),   # R204
    ("Kebersihan R & Nilai S & Lokasi S & Kamar R & Tidur T", "CUKUP"),   # R205
    ("Kebersihan R & Nilai S & Lokasi S & Kamar R & Tidur S", "BURUK"),   # R206
    ("Kebersihan R & Nilai S & Lokasi S & Kamar R & Tidur R", "BURUK"),   # R207
    # C-R V-S L-R
    ("Kebersihan R & Nilai S & Lokasi R & Kamar T & Tidur T", "CUKUP"),   # R208
    ("Kebersihan R & Nilai S & Lokasi R & Kamar T & Tidur S", "CUKUP"),   # R209
    ("Kebersihan R & Nilai S & Lokasi R & Kamar T & Tidur R", "BURUK"),   # R210
    ("Kebersihan R & Nilai S & Lokasi R & Kamar S & Tidur T", "CUKUP"),   # R211
    ("Kebersihan R & Nilai S & Lokasi R & Kamar S & Tidur S", "BURUK"),   # R212
    ("Kebersihan R & Nilai S & Lokasi R & Kamar S & Tidur R", "BURUK"),   # R213
    ("Kebersihan R & Nilai S & Lokasi R & Kamar R & Tidur T", "BURUK"),   # R214
    ("Kebersihan R & Nilai S & Lokasi R & Kamar R & Tidur S", "BURUK"),   # R215
    ("Kebersihan R & Nilai S & Lokasi R & Kamar R & Tidur R", "BURUK"),   # R216
    # C-R V-R L-T
    ("Kebersihan R & Nilai R & Lokasi T & Kamar T & Tidur T", "CUKUP"),   # R217
    ("Kebersihan R & Nilai R & Lokasi T & Kamar T & Tidur S", "CUKUP"),   # R218
    ("Kebersihan R & Nilai R & Lokasi T & Kamar T & Tidur R", "CUKUP"),   # R219
    ("Kebersihan R & Nilai R & Lokasi T & Kamar S & Tidur T", "CUKUP"),   # R220
    ("Kebersihan R & Nilai R & Lokasi T & Kamar S & Tidur S", "CUKUP"),   # R221
    ("Kebersihan R & Nilai R & Lokasi T & Kamar S & Tidur R", "BURUK"),   # R222
    ("Kebersihan R & Nilai R & Lokasi T & Kamar R & Tidur T", "CUKUP"),   # R223
    ("Kebersihan R & Nilai R & Lokasi T & Kamar R & Tidur S", "BURUK"),   # R224
    ("Kebersihan R & Nilai R & Lokasi T & Kamar R & Tidur R", "BURUK"),   # R225
    # C-R V-R L-S
    ("Kebersihan R & Nilai R & Lokasi S & Kamar T & Tidur T", "CUKUP"),   # R226
    ("Kebersihan R & Nilai R & Lokasi S & Kamar T & Tidur S", "CUKUP"),   # R227
    ("Kebersihan R & Nilai R & Lokasi S & Kamar T & Tidur R", "BURUK"),   # R228
    ("Kebersihan R & Nilai R & Lokasi S & Kamar S & Tidur T", "CUKUP"),   # R229
    ("Kebersihan R & Nilai R & Lokasi S & Kamar S & Tidur S", "BURUK"),   # R230
    ("Kebersihan R & Nilai R & Lokasi S & Kamar S & Tidur R", "BURUK"),   # R231
    ("Kebersihan R & Nilai R & Lokasi S & Kamar R & Tidur T", "BURUK"),   # R232
    ("Kebersihan R & Nilai R & Lokasi S & Kamar R & Tidur S", "BURUK"),   # R233
    ("Kebersihan R & Nilai R & Lokasi S & Kamar R & Tidur R", "BURUK"),   # R234
    # C-R V-R L-R
    ("Kebersihan R & Nilai R & Lokasi R & Kamar T & Tidur T", "CUKUP"),   # R235
    ("Kebersihan R & Nilai R & Lokasi R & Kamar T & Tidur S", "BURUK"),   # R236
    ("Kebersihan R & Nilai R & Lokasi R & Kamar T & Tidur R", "BURUK"),   # R237
    ("Kebersihan R & Nilai R & Lokasi R & Kamar S & Tidur T", "BURUK"),   # R238
    ("Kebersihan R & Nilai R & Lokasi R & Kamar S & Tidur S", "BURUK"),   # R239
    ("Kebersihan R & Nilai R & Lokasi R & Kamar S & Tidur R", "BURUK"),   # R240
    ("Kebersihan R & Nilai R & Lokasi R & Kamar R & Tidur T", "BURUK"),   # R241
    ("Kebersihan R & Nilai R & Lokasi R & Kamar R & Tidur S", "BURUK"),   # R242
    ("Kebersihan R & Nilai R & Lokasi R & Kamar R & Tidur R", "BURUK"),   # R243
]


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGASI
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 4px'>
        <div style='font-size:3rem'>🏨</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e8e8e8'>SPK Hotel</div>
        <div style='font-size:0.78rem; color:#aaa; margin-top:2px'>Fuzzy Logic Mamdani</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    halaman = st.radio(
        "Navigasi",
        ["🏠 Beranda", "📊 Dataset", "🔢 Hitung SPK", "📈 Visualisasi", "👥 Profil Kelompok"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#888; text-align:center; padding:4px'>
        SCPK — UPN "Veteran" Yogyakarta<br>TA 2025/2026
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN 1 — BERANDA
# ══════════════════════════════════════════════════════════════
if halaman == "🏠 Beranda":
    st.title("🏨 Sistem Pendukung Keputusan Rekomendasi Hotel")
    st.markdown("#### Metode Fuzzy Logic — Mamdani  |  SCPK 2025/2026")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Data",    f"{len(df):,}")
    c2.metric("📋 Kriteria",      "5 Kriteria")
    c3.metric("📏 Rule Base",     "243 Aturan")
    c4.metric("⚙️ Metode",        "Fuzzy Mamdani")

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        ### 📌 Tentang Sistem
        Sistem ini membantu pengguna **merekomendasikan hotel** berdasarkan ulasan
        tamu menggunakan **Fuzzy Logic Mamdani dengan Pembobotan Preferensi Pengguna**.  
        Output berupa **skor 0–100** dan label rekomendasi otomatis yang dipengaruhi
        oleh kualitas hotel (fuzzy) dan prioritas pengguna (bobot).

        ### 🔍 Kriteria Penilaian
        | # | Kriteria | Skala | Keterangan |
        |---|----------|-------|------------|
        | 1 | Kebersihan (Cleanliness) | 1–5 | Kebersihan fasilitas |
        | 2 | Nilai/Harga (Value) | 1–5 | Kesesuaian harga & fasilitas |
        | 3 | Lokasi (Location) | 1–5 | Aksesibilitas & posisi strategis |
        | 4 | Kamar (Rooms) | 1–5 | Kenyamanan kamar |
        | 5 | Kualitas Tidur (Sleep Quality) | 1–5 | Ketenangan & kualitas tidur |
        """)
    with col_b:
        st.markdown("""
        ### ⚙️ Alur Proses Fuzzy + Bobot
        """)
        st.markdown("""
        ```
        Input Nilai (1–5)
              ↓
        ┌─────────────────────┐
        │  TAHAP 1: FUZZY     │
        │  Fuzzifikasi        │
        │  Inferensi Mamdani  │
        │  Defuzzifikasi      │
        │  → fuzzy_score      │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │  TAHAP 2: BOBOT     │
        │  User slider bobot  │
        │  Normalisasi bobot  │
        │  Weighted average   │
        │  → pref_score       │
        └─────────────────────┘
              ↓
        final = 0.7×fuzzy
              + 0.3×pref
              ↓
        Skor (0–100) + Ranking
        ```
        """)

    st.markdown("---")
    st.markdown("### 🏷️ Label Rekomendasi")
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.success("✅ **Direkomendasikan**\nSkor ≥ 65")
    col_l2.warning("⚠️ **Cukup**\n40 ≤ Skor < 65")
    col_l3.error("❌ **Tidak Direkomendasikan**\nSkor < 40")


# ══════════════════════════════════════════════════════════════
# HALAMAN 2 — DATASET
# ══════════════════════════════════════════════════════════════
elif halaman == "📊 Dataset":
    st.title("📊 Dataset Hotel Reviews")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Baris", f"{len(df):,}")
    c2.metric("📋 Kolom",       f"{len(df.columns)}")
    c3.metric("🌐 Sumber",      "CSV (Hugging Face)")
    c4.metric("🎯 Tema",        "Pariwisata")

    st.markdown("### 📋 Tabel Dataset")
    kolom_tampil = ["hotel_id", "title", "cleanliness", "value",
                    "location", "rooms", "sleep_quality", "overall"]
    avail = [c for c in kolom_tampil if c in df.columns]
    df_view = df[avail].rename(columns={
        "hotel_id": "ID Hotel", "title": "Judul Ulasan",
        "cleanliness": "Kebersihan", "value": "Nilai",
        "location": "Lokasi", "rooms": "Kamar",
        "sleep_quality": "Kualitas Tidur", "overall": "Overall"
    })
    # Format angka agar bersih
    for col in ["Kebersihan", "Nilai", "Lokasi", "Kamar", "Kualitas Tidur", "Overall"]:
        if col in df_view.columns:
            df_view[col] = df_view[col].apply(
                lambda x: f"{int(x)}" if pd.notna(x) and x == int(x) else f"{x:.1f}"
            )
    st.dataframe(df_view, use_container_width=True, height=420)

    st.markdown("### 📈 Statistik Deskriptif")
    st.dataframe(
        df[["cleanliness", "value", "location", "rooms",
            "sleep_quality", "overall"]].describe().round(2),
        use_container_width=True
    )

    # Distribusi tiap kriteria
    st.markdown("### 📊 Distribusi Nilai Kriteria")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor("#f8f9fa")
    cols_dist = ["cleanliness", "value", "location", "rooms", "sleep_quality", "overall"]
    labels_dist = ["Kebersihan", "Nilai/Harga", "Lokasi",
                   "Kamar", "Kualitas Tidur", "Overall"]
    palette = ["#667eea","#764ba2","#f093fb","#f5576c","#4facfe","#43e97b"]
    for i, (col, lbl, clr) in enumerate(zip(cols_dist, labels_dist, palette)):
        ax = axes[i//3][i%3]
        ax.hist(df[col].dropna(), bins=15, color=clr, edgecolor="white", alpha=0.85)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_xlabel("Nilai"); ax.set_ylabel("Frekuensi")
        ax.grid(True, alpha=0.3); ax.set_facecolor("#ffffff")
    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════
# HALAMAN 3 — HITUNG SPK
# ══════════════════════════════════════════════════════════════
elif halaman == "🔢 Hitung SPK":
    st.title("🔢 Perhitungan SPK — Fuzzy + Bobot Fuzzy Mamdani + Preferensi Pengguna")
    st.markdown("---")

    # ── Penjelasan Metode Fuzzy + Bobot ───────────────────────────────
    st.info("""
ℹ️ **Metode: Fuzzy + Bobot Fuzzy Mamdani + User Preference Weighting**

Sistem menggunakan dua tahap perhitungan:
- **Tahap 1 — Fuzzy Mamdani:** Inferensi kualitas hotel berdasarkan 243 rule IF-THEN (3^5 kombinasi penuh) → menghasilkan *Skor Fuzzy*.
- **Tahap 2 — Preference Weighting:** Bobot pengguna menentukan prioritas kriteria → menghasilkan *Skor Preferensi*.
- **Skor Akhir = 0.7 × Skor Fuzzy + 0.3 × Skor Preferensi** — mengubah ranking sesuai prioritas user.
    """)

    # ── Input Bobot Preferensi ─────────────────────────────────
    st.markdown("### ⚖️ Bobot Preferensi Pengguna")
    st.caption("Sesuaikan prioritas setiap kriteria (1 = rendah, 10 = sangat penting). Bobot dinormalisasi otomatis.")

    col1, col2, col3 = st.columns(3)
    with col1:
        w_clean = st.slider("🧹 Kebersihan",     1, 10, 5, key="w1")
        w_value = st.slider("💰 Nilai/Harga",    1, 10, 4, key="w2")
    with col2:
        w_loc   = st.slider("📍 Lokasi",         1, 10, 4, key="w3")
        w_rooms = st.slider("🛏️ Kamar",          1, 10, 4, key="w4")
    with col3:
        w_sleep = st.slider("😴 Kualitas Tidur", 1, 10, 3, key="w5")

    # Normalisasi bobot
    n_c, n_v, n_l, n_r, n_s = normalisasi_bobot(w_clean, w_value, w_loc, w_rooms, w_sleep)

    # Tampilkan bobot ternormalisasi
    df_bobot = pd.DataFrame({
        "Kriteria":     ["🧹 Kebersihan", "💰 Nilai/Harga", "📍 Lokasi", "🛏️ Kamar", "😴 Kualitas Tidur"],
        "Bobot Input":  [w_clean, w_value, w_loc, w_rooms, w_sleep],
        "Bobot Normal": [f"{v:.4f}" for v in [n_c, n_v, n_l, n_r, n_s]],
        "Persentase":   [f"{v*100:.1f}%" for v in [n_c, n_v, n_l, n_r, n_s]],
    })
    st.dataframe(df_bobot.set_index("Kriteria"), use_container_width=True)

    st.markdown("---")

    # ── Opsi Perhitungan ───────────────────────────────────────
    st.markdown("### 🔧 Opsi Perhitungan")
    col_a, col_b = st.columns(2)
    with col_a:
        n_data = st.number_input(
            "Jumlah data yang dihitung",
            min_value=10, max_value=len(df),
            value=min(250, len(df)), step=10
        )
    with col_b:
        min_skor_filter = st.selectbox(
            "Filter Tampil Hasil",
            ["Semua", "Hanya Direkomendasikan", "Hanya Cukup", "Hanya Tidak Direkomendasikan"]
        )

    st.markdown("---")

    # ── Tombol Hitung ──────────────────────────────────────────
    if st.button("🚀 Hitung SPK Fuzzy + Bobot", type="primary", use_container_width=True):
        df_hitung  = df.head(int(n_data)).copy()
        progress   = st.progress(0, text="Memulai perhitungan fuzzy...")
        list_fuzzy, list_pref, list_final = [], [], []

        for i, (_, row) in enumerate(df_hitung.iterrows()):
            fs = hitung_fuzzy(row)
            ps = hitung_pref_score(row, n_c, n_v, n_l, n_r, n_s)
            fn = hitung_final_score(fs, ps)
            list_fuzzy.append(fs)
            list_pref.append(ps)
            list_final.append(fn)
            if (i + 1) % max(1, int(n_data) // 20) == 0:
                pct = int((i + 1) / int(n_data) * 100)
                progress.progress(min(pct, 99), text=f"Menghitung... {i+1}/{int(n_data)}")

        progress.progress(100, text="✅ Selesai!")

        df_hitung["Skor Fuzzy"]  = list_fuzzy
        df_hitung["Skor Pref"]   = list_pref
        df_hitung["Skor Akhir"]  = list_final
        df_hitung["Rekomendasi"] = df_hitung["Skor Akhir"].apply(label_rekomendasi)

        # Evaluasi kedekatan vs overall
        if "overall" in df_hitung.columns:
            df_hitung["Overall_100"] = (df_hitung["overall"] * 20).clip(0, 100).round(1)
            df_hitung["Error_ABS"]   = (
                df_hitung["Skor Akhir"] - df_hitung["Overall_100"]
            ).abs().round(2)

        df_hitung = df_hitung.dropna(subset=["Skor Akhir"])
        df_hitung = df_hitung.sort_values("Skor Akhir", ascending=False).reset_index(drop=True)
        df_hitung.index += 1
        df_hitung.index.name = "Peringkat"

        # Simpan bobot aktif ke session state
        st.session_state["hasil"] = df_hitung
        st.session_state["bobot_aktif"] = {
            "Kebersihan": f"{n_c*100:.1f}%",
            "Nilai/Harga": f"{n_v*100:.1f}%",
            "Lokasi": f"{n_l*100:.1f}%",
            "Kamar": f"{n_r*100:.1f}%",
            "Kualitas Tidur": f"{n_s*100:.1f}%",
        }
        st.rerun()

    # ── Tampilkan Hasil ────────────────────────────────────────
    if "hasil" in st.session_state:
        df_res = st.session_state["hasil"].copy()

        # ── Info Bobot Aktif ───────────────────────────────────
        if "bobot_aktif" in st.session_state:
            ba = st.session_state["bobot_aktif"]
            st.markdown("#### 🎯 Bobot Preferensi Aktif")
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            bc1.metric("🧹 Kebersihan",    ba["Kebersihan"])
            bc2.metric("💰 Nilai/Harga",   ba["Nilai/Harga"])
            bc3.metric("📍 Lokasi",        ba["Lokasi"])
            bc4.metric("🛏️ Kamar",         ba["Kamar"])
            bc5.metric("😴 Kualitas Tidur", ba["Kualitas Tidur"])
            st.markdown("---")

        # Filter
        if min_skor_filter == "Hanya Direkomendasikan":
            df_res = df_res[df_res["Rekomendasi"] == "✅ Direkomendasikan"]
        elif min_skor_filter == "Hanya Cukup":
            df_res = df_res[df_res["Rekomendasi"] == "⚠️ Cukup"]
        elif min_skor_filter == "Hanya Tidak Direkomendasikan":
            df_res = df_res[df_res["Rekomendasi"] == "❌ Tidak Direkomendasikan"]

        # ── Ringkasan ──────────────────────────────────────────
        st.markdown("### 📊 Ringkasan Hasil")
        rek = st.session_state["hasil"]["Rekomendasi"]
        sf  = st.session_state["hasil"]["Skor Akhir"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("✅ Direkomendasikan",       (rek == "✅ Direkomendasikan").sum())
        c2.metric("⚠️ Cukup",                 (rek == "⚠️ Cukup").sum())
        c3.metric("❌ Tidak Direkomendasikan", (rek == "❌ Tidak Direkomendasikan").sum())
        c4.metric("📈 Skor Tertinggi",         f"{sf.max():.1f}")
        c5.metric("📉 Skor Rata-rata",         f"{sf.mean():.1f}")

        # ── Tabel Hasil ────────────────────────────────────────
        st.markdown("### 🏆 Tabel Perangkingan")
        cols_tampil = ["hotel_id", "cleanliness", "value", "location",
                       "rooms", "sleep_quality", "Skor Fuzzy", "Skor Pref", "Skor Akhir", "Rekomendasi"]
        if "title" in df_res.columns:
            cols_tampil.insert(1, "title")
        avail_cols = [c for c in cols_tampil if c in df_res.columns]

        df_tampil = df_res[avail_cols].rename(columns={
            "hotel_id": "ID Hotel", "title": "Judul Ulasan",
            "cleanliness": "Kebersihan", "value": "Nilai",
            "location": "Lokasi", "rooms": "Kamar",
            "sleep_quality": "Kualitas Tidur",
            "Skor Pref": "Skor Pref.",
        })
        # Format angka bersih
        for col in ["Kebersihan", "Nilai", "Lokasi", "Kamar", "Kualitas Tidur"]:
            if col in df_tampil.columns:
                df_tampil[col] = df_tampil[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) and x == int(x) else f"{x:.1f}"
                )
        for col in ["Skor Fuzzy", "Skor Pref.", "Skor Akhir"]:
            if col in df_tampil.columns:
                df_tampil[col] = df_tampil[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) and x == int(x) else f"{x:.1f}"
                )

        st.dataframe(
            df_tampil.style
                .set_properties(**{"text-align": "center"})
                .set_table_styles([
                    {"selector": "th", "props": [("text-align", "center")]},
                    {"selector": "td", "props": [("text-align", "center")]},
                ]),
            use_container_width=True, height=420
        )

        # ── Top 10 Hotel Terbaik ───────────────────────────────
        st.markdown("---")
        st.markdown("### 🥇 Top 10 Hotel Terbaik")
        top10 = st.session_state["hasil"].head(10).copy()

        fig_top, ax_top = plt.subplots(figsize=(11, 5))
        fig_top.patch.set_facecolor("#f8f9fa")
        idx_list = list(range(len(top10)))
        labels_top = [f"#{i+1}  ID:{r.hotel_id}" for i, r in enumerate(top10.itertuples())]

        scores_fuzzy = top10["Skor Fuzzy"].values[::-1]
        scores_akhir = top10["Skor Akhir"].values[::-1]
        labels_rev   = labels_top[::-1]
        y = np.arange(len(labels_rev))

        bars_f = ax_top.barh(y - 0.2, scores_fuzzy, height=0.38,
                              color="#667eea", alpha=0.8, label="Skor Fuzzy", edgecolor="white")
        bars_a = ax_top.barh(y + 0.2, scores_akhir, height=0.38,
                              color="#27ae60", alpha=0.85, label="Skor Akhir (Fuzzy + Bobot)", edgecolor="white")

        for bar, val in zip(bars_a, scores_akhir):
            ax_top.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}", va="center", fontsize=8, fontweight="bold", color="#27ae60")

        ax_top.set_yticks(y)
        ax_top.set_yticklabels(labels_rev, fontsize=8)
        ax_top.set_xlabel("Skor (0–100)")
        ax_top.set_title("Top 10 Hotel — Skor Fuzzy vs Skor Akhir Fuzzy + Bobot",
                          fontsize=12, fontweight="bold")
        ax_top.axvline(65, color="#e74c3c", linestyle="--", alpha=0.55,
                        linewidth=1.2, label="Batas Rekomendasikan (65)")
        ax_top.set_xlim(0, 110)
        ax_top.legend(fontsize=8, loc="lower right")
        ax_top.grid(axis="x", alpha=0.3)
        ax_top.set_facecolor("#ffffff")
        plt.tight_layout()
        st.pyplot(fig_top)
        plt.close()

        # ── Detail Peringkat 1 ─────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Detail Hotel Peringkat 1")
        top = st.session_state["hasil"].iloc[0]
        st.success(
            f"🥇 Hotel terbaik: ID **{top['hotel_id']}** | "
            f"Skor Akhir: **{top['Skor Akhir']:.1f}** | "
            f"(Fuzzy: {top['Skor Fuzzy']:.1f} | Pref: {top['Skor Pref']:.1f}) | "
            f"{top['Rekomendasi']}"
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**📥 Input Nilai:**")
            for label, key in [("Kebersihan","cleanliness"),("Nilai/Harga","value"),
                                ("Lokasi","location"),("Kamar","rooms"),
                                ("Kualitas Tidur","sleep_quality")]:
                st.markdown(f"- {label}: **{top[key]:.1f}** / 5")
        with col_d2:
            st.markdown("**⚙️ Proses Fuzzy + Bobot:**")
            st.markdown("- Metode Inferensi    : **Mamdani**")
            st.markdown("- Fungsi Keanggotaan  : **trimf (rendah/tinggi) + trapmf (sedang)**")
            st.markdown("- Defuzzifikasi       : **Centroid**")
            st.markdown(f"- Skor Fuzzy (τ₁)    : **{top['Skor Fuzzy']:.1f}** / 100")
            st.markdown(f"- Skor Preferensi (τ₂): **{top['Skor Pref']:.1f}** / 100")
            st.markdown(f"- Skor Akhir          : **0.7×{top['Skor Fuzzy']:.1f} + 0.3×{top['Skor Pref']:.1f} = {top['Skor Akhir']:.1f}**")
            st.markdown(f"- Rekomendasi         : **{top['Rekomendasi']}**")

        # ── Evaluasi Kedekatan vs Overall ──────────────────────
        if "Error_ABS" in st.session_state["hasil"].columns:
            st.markdown("---")
            st.markdown("### 📐 Evaluasi Kedekatan Hasil — Skor Akhir vs Rating Pengguna")

            df_eval = st.session_state["hasil"].dropna(subset=["Error_ABS"])
            mae     = df_eval["Error_ABS"].mean()
            corr    = df_eval["Skor Akhir"].corr(df_eval["Overall_100"])

            ce1, ce2, ce3 = st.columns(3)
            ce1.metric("📉 MAE",              f"{mae:.2f} poin")
            ce2.metric("🔗 Korelasi Pearson",  f"{corr:.3f}")
            ce3.metric("📊 Data Dievaluasi",   f"{len(df_eval):,} baris")

            kualitas = (
                "sangat baik — sistem sangat dekat dengan penilaian pengguna" if mae <= 8
                else "baik — sistem cukup dekat dengan penilaian pengguna" if mae <= 15
                else "moderat — selisih yang wajar antara inferensi sistem dan rating pengguna"
            )
            kat_corr = (
                "korelasi positif kuat" if corr >= 0.7
                else "korelasi positif sedang" if corr >= 0.4
                else "korelasi lemah"
            )
            st.info(f"""
**Catatan Evaluasi Akademis:**
Evaluasi dilakukan dengan membandingkan *Skor Akhir* sistem terhadap nilai *overall* pada dataset
untuk melihat tingkat kedekatan sistem dengan penilaian pengguna.
Kolom *overall* dikonversi ke skala 0–100 (×20) agar sebanding dengan output sistem.

- **MAE = {mae:.2f} poin** → Kualitas sistem **{kualitas}**.
- **Korelasi Pearson = {corr:.3f}** → Terdapat **{kat_corr}** antara skor sistem dan rating pengguna.
- Nilai *overall* merupakan rating subjektif pengguna, bukan ground truth absolut,
  sehingga selisih tertentu merupakan hal yang wajar secara metodologis.
- Bobot preferensi aktif memengaruhi *Skor Akhir* — mengubah bobot akan menggeser ranking hotel.
            """)

            fig_eval, ax_e = plt.subplots(figsize=(7, 5))
            fig_eval.patch.set_facecolor("#f8f9fa")
            ax_e.scatter(df_eval["Overall_100"], df_eval["Skor Akhir"],
                         alpha=0.45, color="#667eea", s=18, edgecolors="none",
                         label="Data Observasi (Skor Akhir)")
            lims = [0, 100]
            ax_e.plot(lims, lims, "r--", linewidth=1.2, label="Garis Ideal (y = x)")
            ax_e.set_xlabel("Rating Pengguna Overall × 20  (Skala 0–100)", fontsize=10)
            ax_e.set_ylabel("Skor Akhir Fuzzy + Bobot  (Skala 0–100)", fontsize=10)
            ax_e.set_title("Kedekatan Skor Akhir terhadap Rating Pengguna",
                            fontweight="bold", fontsize=11)
            ax_e.legend(fontsize=9)
            ax_e.grid(True, alpha=0.3)
            ax_e.set_facecolor("#ffffff")
            plt.tight_layout()
            st.pyplot(fig_eval)
            plt.close()

        # ── Rule Base ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📜 Rule Base — 243 Aturan Penuh (3^5 Kombinasi)")
        st.caption("Keterangan: **T**=Tinggi | **S**=Sedang | **R**=Rendah  ·  Total: 243 rule dari 3^5 = 3×3×3×3×3 kombinasi")

        n_bagus = sum(1 for r in RULES_LIST if r[1] == "BAGUS")
        n_cukup = sum(1 for r in RULES_LIST if r[1] == "CUKUP")
        n_buruk = sum(1 for r in RULES_LIST if r[1] == "BURUK")
        rb1, rb2, rb3, rb4 = st.columns(4)
        rb1.metric("📋 Total Rule",  f"{len(RULES_LIST)}")
        rb2.metric("✅ → BAGUS",     f"{n_bagus}")
        rb3.metric("⚠️ → CUKUP",     f"{n_cukup}")
        rb4.metric("❌ → BURUK",     f"{n_buruk}")

        rules_df = pd.DataFrame({
            "No":               [f"R{i}" for i in range(1, len(RULES_LIST)+1)],
            "IF (Antecedent)":  [r[0] for r in RULES_LIST],
            "THEN (Consequent)":[f"Skor {r[1]}" for r in RULES_LIST],
        })
        st.dataframe(
            rules_df.set_index("No").style
                .set_properties(**{"text-align": "left"})
                .apply(lambda x: [
                    "background-color: #d5f5e3; color:black;" if "BAGUS" in v
                    else "background-color: #fef9e7; color:black;" if "CUKUP" in v
                    else "background-color: #fadbd8; color:black;" if "BURUK" in v
                    else "" for v in x
                ], subset=["THEN (Consequent)"]),
            use_container_width=True, height=620
        )

    else:
        st.info("⬆️ Atur bobot preferensi dan opsi perhitungan di atas, lalu klik **🚀 Hitung SPK Fuzzy + Bobot** untuk memulai.")




# ══════════════════════════════════════════════════════════════
# HALAMAN 4 — VISUALISASI
# ══════════════════════════════════════════════════════════════
elif halaman == "📈 Visualisasi":
    st.title("📈 Visualisasi Fuzzy & Hasil")
    st.markdown("---")

    # ── Membership Function ────────────────────────────────────
    st.markdown("### 🔷 Fungsi Keanggotaan Variabel Input & Output")

    fig_mf, axes_mf = plt.subplots(2, 3, figsize=(16, 9))
    fig_mf.patch.set_facecolor("#f0f2f6")
    fig_mf.suptitle("Fungsi Keanggotaan Fuzzy Mamdani — SPK Rekomendasi Hotel",
                    fontsize=13, fontweight="bold", color="#2c3e50", y=1.01)

    vars_info = [
        (fz_clean, "Kebersihan (Cleanliness)", False),
        (fz_value, "Nilai/Harga (Value)",      False),
        (fz_loc,   "Lokasi (Location)",        False),
        (fz_rooms, "Kamar (Rooms)",            False),
        (fz_sleep, "Kualitas Tidur (Sleep Q.)",False),
        (fz_skor,  "Skor Output",              True),
    ]

    # Warna & label per himpunan
    clr_in   = {"rendah": "#e74c3c", "sedang": "#e67e22", "tinggi": "#27ae60"}
    clr_out  = {"buruk":  "#e74c3c", "cukup":  "#e67e22", "bagus":  "#27ae60"}
    lbl_in   = {"rendah": "Rendah", "sedang": "Sedang", "tinggi": "Tinggi"}
    lbl_out  = {"buruk":  "Buruk",  "cukup":  "Cukup",  "bagus":  "Bagus"}
    ls_map   = {"rendah": "-", "sedang": "--", "tinggi": "-.",
                "buruk":  "-", "cukup":  "--", "bagus":  "-."}

    for idx, (var, title, is_out) in enumerate(vars_info):
        ax   = axes_mf[idx // 3][idx % 3]
        cmap = clr_out if is_out else clr_in
        lmap = lbl_out if is_out else lbl_in

        for term in var.terms:
            mf_vals = fuzz.interp_membership(var.universe, var[term].mf, var.universe)
            col = cmap.get(term, "#3498db")
            ax.plot(var.universe, mf_vals,
                    label=lmap.get(term, term.capitalize()),
                    color=col,
                    linewidth=2.4,
                    linestyle=ls_map.get(term, "-"),
                    zorder=3)
            ax.fill_between(var.universe, mf_vals, alpha=0.10,
                            color=col, zorder=2)

        # Styling per panel
        ax.set_facecolor("#ffffff")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#2c3e50", pad=8)
        ax.set_xlabel("Skor (0–100)" if is_out else "Nilai Input (1–5)",
                      fontsize=8.5, color="#555")
        ax.set_ylabel("μ — Derajat Keanggotaan", fontsize=8.5, color="#555")
        ax.set_ylim(-0.05, 1.22)
        if is_out:
            ax.set_xlim(0, 100)
        else:
            ax.set_xlim(1, 5)
            ax.set_xticks([1, 2, 3, 4, 5])
        ax.tick_params(labelsize=8)
        ax.axhline(y=0, color="#ccc", linewidth=0.8, zorder=1)
        ax.axhline(y=1, color="#bbb", linewidth=0.6, linestyle=":", zorder=1)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
        leg = ax.legend(fontsize=8, loc="upper right",
                        framealpha=0.88, edgecolor="#ccc",
                        fancybox=True, borderpad=0.6)
        for line in leg.get_lines():
            line.set_linewidth(2.0)
        # Bingkai tipis
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.8)

    plt.tight_layout(pad=2.8)
    st.pyplot(fig_mf)
    plt.close()

    # ── Visualisasi Hasil (jika sudah dihitung) ────────────────
    if "hasil" in st.session_state:
        df_h = st.session_state["hasil"]
        st.markdown("---")
        st.markdown("### 📊 Distribusi Hasil Perhitungan SPK")

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig2.patch.set_facecolor("#f8f9fa")

        # Pie chart
        rek_counts  = df_h["Rekomendasi"].value_counts()
        label_clean = [r.replace("✅ ","").replace("⚠️ ","").replace("❌ ","")
                       for r in rek_counts.index]
        pie_colors  = [
            "#27ae60" if "Direk" in r else "#f39c12" if "Cukup" in r else "#e74c3c"
            for r in rek_counts.index
        ]
        axes2[0].pie(rek_counts.values, labels=label_clean,
                     autopct="%1.1f%%", colors=pie_colors,
                     startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
        axes2[0].set_title("Distribusi Rekomendasi", fontweight="bold")

        # Histogram
        axes2[1].hist(df_h["Skor Fuzzy"].dropna(), bins=20,
                      color="#667eea", edgecolor="white", alpha=0.85)
        axes2[1].axvline(40, color="#f39c12", linestyle="--", linewidth=1.5,
                          label="Batas Cukup (40)")
        axes2[1].axvline(65, color="#27ae60", linestyle="--", linewidth=1.5,
                          label="Batas Rekomendasikan (65)")
        axes2[1].set_xlabel("Skor Fuzzy")
        axes2[1].set_ylabel("Frekuensi")
        axes2[1].set_title("Histogram Skor Fuzzy", fontweight="bold")
        axes2[1].legend(fontsize=8)
        axes2[1].grid(True, alpha=0.3)
        axes2[1].set_facecolor("#ffffff")

        # Scatter Overall vs Fuzzy
        if "Overall_100" in df_h.columns:
            axes2[2].scatter(df_h["Overall_100"], df_h["Skor Fuzzy"],
                             alpha=0.4, color="#764ba2", s=15, edgecolors="none",
                             label="Data Observasi")
            axes2[2].plot([0,100],[0,100], "r--", linewidth=1.2, label="Garis Ideal (y=x)")
            axes2[2].set_xlabel("Rating Pengguna × 20")
            axes2[2].set_ylabel("Skor Fuzzy Mamdani")
            axes2[2].set_title("Kedekatan Fuzzy vs Rating Pengguna", fontweight="bold")
            axes2[2].legend(fontsize=8)
            axes2[2].grid(True, alpha=0.3)
            axes2[2].set_facecolor("#ffffff")
        else:
            axes2[2].axis("off")

        plt.tight_layout(pad=2)
        st.pyplot(fig2)
        plt.close()

        # ── Grafik Error ────────────────────────────────────────
        if "Error_ABS" in df_h.columns:
            st.markdown("---")
            st.markdown("### 📉 Distribusi Selisih Absolut | Skor Fuzzy – Rating Pengguna × 20 |")
            fig_err, ax_err = plt.subplots(figsize=(10, 4))
            fig_err.patch.set_facecolor("#f8f9fa")
            ax_err.hist(df_h["Error_ABS"].dropna(), bins=25,
                        color="#f5576c", edgecolor="white", alpha=0.85)
            ax_err.axvline(df_h["Error_ABS"].mean(), color="#1a1a2e",
                            linestyle="--", linewidth=1.8,
                            label=f"MAE = {df_h['Error_ABS'].mean():.2f}")
            ax_err.set_xlabel("Selisih Absolut (poin)"); ax_err.set_ylabel("Frekuensi")
            ax_err.set_title("Distribusi Selisih Absolut Skor Fuzzy vs Rating Pengguna",
                              fontweight="bold")
            ax_err.legend(); ax_err.grid(True, alpha=0.3)
            ax_err.set_facecolor("#ffffff")
            plt.tight_layout()
            st.pyplot(fig_err)
            plt.close()
    else:
        st.info("💡 Belum ada hasil. Silakan ke menu **🔢 Hitung SPK** dan jalankan perhitungan terlebih dahulu.")


# ══════════════════════════════════════════════════════════════
# HALAMAN 5 — PROFIL KELOMPOK
# ══════════════════════════════════════════════════════════════
elif halaman == "👥 Profil Kelompok":
    st.title("👥 Profil Kelompok")
    st.markdown("---")

    st.markdown("""
    ### 🏫 Informasi Mata Kuliah
    | | |
    |---|---|
    | **Mata Kuliah** | Sistem Cerdas & Pendukung Keputusan (SCPK) |
    | **Tahun Akademik** | 2025/2026 |
    | **Tema** | Pariwisata & Hospitality |
    | **Metode** | Fuzzy Logic (Mamdani) |
    | **Program Studi** | Informatika — UPN "Veteran" Yogyakarta |

    ---
    ### 👤 Anggota Kelompok
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Anggota 1
        - **Nama :** *Kevin Prasetya*
        - **NIM  :** *123240231*
        """)
    with col2:
        st.markdown("""
        #### Anggota 2
        - **Nama :** *Nur Aziz Ramamdhan*
        - **NIM  :** *123240261*
        """)

    st.markdown("---")
    st.markdown("""
    ### 📁 Judul Proyek
    > **"Sistem Pendukung Keputusan Rekomendasi Hotel Berbasis Fuzzy Logic Mamdani
    > untuk Pariwisata & Hospitality"**

    ### 📦 Dataset
    | | |
    |---|---|
    | **Sumber** | Hugging Face / Online |
    | **Format** | CSV (`data_baru.csv`) |
    | **Kriteria** | Cleanliness, Value, Location, Rooms, Sleep Quality |
    | **Output** | Skor Fuzzy 0–100 + Label Rekomendasi |

    ### 🧪 Spesifikasi Teknis
    | | |
    |---|---|
    | **Metode** | Fuzzy Mamdani |
    | **Fungsi Keanggotaan** | trimf (rendah/tinggi) + trapmf (sedang/cukup) |
    | **Defuzzifikasi** | Centroid |
    | **Jumlah Rule** | 243 Aturan (3^5 Kombinasi Penuh) |
    | **Himpunan Fuzzy Input** | Rendah / Sedang / Tinggi |
    | **Himpunan Fuzzy Output** | Buruk / Cukup / Bagus |
    | **Framework** | Python · Streamlit · scikit-fuzzy |
    """)