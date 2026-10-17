from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data_set" / "securities.csv"

print("=" * 70)
print("📊 EDA SPECIFIQUE : securities.csv")
print("=" * 70)

df = pd.read_csv(FILE_PATH)

# 1. Overview
print("\n[ 1. OVERVIEW ]")
print(f"• Nombre d'entreprises répertoriées : {len(df)}")
print(f"• Colonnes disponibles : {list(df.columns)}")
print(f"• Colonnes avec valeurs manquantes :")
missing = df.isna().sum()
print(missing[missing > 0])

# 2. Analyse des secteurs
sector_col = 'GICS Sector'
print(f"\n[ 2. REPARTITION PAR SECTEUR ({sector_col}) ]")
sector_counts = df[sector_col].value_counts()
print(sector_counts)

# 3. Visualisation : Barplot des secteurs
plt.figure(figsize=(10, 5))
sns.barplot(y=sector_counts.index, x=sector_counts.values, hue=sector_counts.index, palette='viridis', legend=False)
plt.title("Nombre d'entreprises S&P 500 par secteur GICS")
plt.xlabel("Nombre d'entreprises")
plt.ylabel("Secteur")
plt.tight_layout()
plt.show()