from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data_set" / "fundamentals.csv"

print("=" * 70)
print("📊 EDA SPECIFIQUE : fundamentals.csv")
print("=" * 70)

df = pd.read_csv(FILE_PATH)

# 1. Overview
print("\n[ 1. OVERVIEW ]")
print(f"• Shape : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"• Doublons : {df.duplicated().sum()}")
print(f"• Total valeurs manquantes : {df.isna().sum().sum()} ({df.isna().sum().sum() / df.size * 100:.2f}%)")

# 2. Quality Alerts (Missing values)
print("\n[ 2. DIAGNOSTIC VALEURS MANQUANTES ]")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    for col, count in missing.items():
        print(f"⚠️ {col} : {count} manquant(s) ({count / len(df) * 100:.2f}%)")

# 3. Analyse Cible (ex: Total Revenue)
target = 'Total Revenue'
print(f"\n[ 3. STATISTIQUES CIBLE : {target} ]")
print(df[target].describe().apply(lambda x: f"{x:,.2f}"))

# 4. Visualisations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution Target
sns.histplot(df[target].dropna(), bins=40, kde=True, ax=axes[0], color='teal')
axes[0].set_title(f'Distribution de {target}')
axes[0].set_xlabel('USD ($)')

# Top 10 Correlations
num_df = df.select_dtypes(include=[np.number])
top_corr = num_df.corr()[target].abs().sort_values(ascending=False).head(10)
sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index, palette='Blues_r', legend=False, ax=axes[1])
axes[1].set_title(f'Top 10 des variables corrélées avec {target}')
axes[1].set_xlabel('Corrélation Absolue')

plt.tight_layout()
plt.show()