import os
from pathlib import Path
import pandas as pd
from ydata_profiling import ProfileReport

# Détection dynamique du dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_set"
OUTPUT_DIR = BASE_DIR / "exploratory" / "reports"

files = {
    "fundamentals": DATA_DIR / "fundamentals.csv",
    "prices": DATA_DIR / "prices.csv",
    "prices_split_adjusted": DATA_DIR / "prices-split-adjusted.csv",
    "securities": DATA_DIR / "securities.csv"
}

# Création du dossier des rapports s'il n'existe pas
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name, filepath in files.items():
    print(f"⏳ Génération du rapport Profiling pour : {name}...")
    if not filepath.exists():
        print(f"❌ Fichier introuvable : {filepath}\n")
        continue

    df = pd.read_csv(filepath)

    profile = ProfileReport(
        df,
        title=f"EDA Profile Report - {name}",
        explorative=True
    )

    output_file = OUTPUT_DIR / f"{name}_report.html"
    profile.to_file(output_file)
    print(f"✅ Rapport enregistré : {output_file}\n")