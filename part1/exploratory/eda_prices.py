from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data_set" / "prices.csv"

print("=" * 70)
print("📊 EDA SPECIFIQUE : prices.csv")
print("=" * 70)

df = pd.read_csv(FILE_PATH)

# 1. Overview
print("\n[ 1. OVERVIEW ]")
print(f"• Shape : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
print(f"• Plage de dates : de {df['date'].min()} à {df['date'].max()}")
print(f"• Nombre d'entreprises (symboles uniques) : {df['symbol'].nunique()}")
print(f"• Doublons exacts : {df.duplicated().sum()}")
print(f"• Valeurs manquantes : {df.isna().sum().sum()}")

# 2. Statistiques descriptives
print("\n[ 2. RESUME DES PRIX & VOLUMES ]")
numeric_cols = ['open', 'close', 'low', 'high', 'volume']
print(df[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])

# 3. Quality Alerts
zeros_vol = (df['volume'] == 0).sum()
print(f"\n[ 3. QUALITY ALERTS ]")
print(f"• Lignes avec Volume = 0 : {zeros_vol} ({zeros_vol / len(df) * 100:.4f}%)")

# 4. Visualisations
plt.figure(figsize=(10, 4))

# Heatmap des corrélations fortes
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
plt.title("Matrice de Corrélation - prices.csv")
plt.tight_layout()
plt.show()