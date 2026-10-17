from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data_set" / "prices-split-adjusted.csv"

print("=" * 70)
print("📊 EDA SPECIFIQUE : prices-split-adjusted.csv")
print("=" * 70)

df = pd.read_csv(FILE_PATH)
df['date'] = pd.to_datetime(df['date'])

# 1. Overview
print("\n[ 1. OVERVIEW ]")
print(f"• Shape : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
print(f"• Date début : {df['date'].min().strftime('%Y-%m-%d')}")
print(f"• Date fin   : {df['date'].max().strftime('%Y-%m-%d')}")
print(f"• Valeurs manquantes : {df.isna().sum().sum()}")

# 2. Analyse temporelle sur une entreprise (Exemple : AAPL)
symbol_sample = 'AAPL'
sample_df = df[df['symbol'] == symbol_sample].sort_values('date')

print(f"\n[ 2. ANATOMIE DE L'ACTION {symbol_sample} ]")
print(f"• Total jours de cotation : {len(sample_df)}")
print(f"• Prix Min / Max (Close Ajusté) : ${sample_df['close'].min():.2f} / ${sample_df['close'].max():.2f}")

# 3. Visualisation : Série temporelle & Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Série temporelle AAPL
axes[0].plot(sample_df['date'], sample_df['close'], color='darkgreen', linewidth=1.5)
axes[0].set_title(f"Évolution du prix de clôture ajusté ({symbol_sample})")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Prix ($)")
axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot Distribution Close Global
sns.histplot(df['close'], bins=50, kde=True, ax=axes[1], color='darkblue')
axes[1].set_title("Distribution globale des prix 'close' (Tous symboles)")
axes[1].set_xlabel("Prix ($)")

plt.tight_layout()
plt.show()