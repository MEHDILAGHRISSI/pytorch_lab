from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Définition dynamique du chemin d'accès relatif au script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "dataset" / "predictive_maintenance.csv"


def preprocess_predictive_maintenance(
    df: pd.DataFrame,
    target_type: str = "multiclass"
) -> tuple:
    """
    Nettoie, génère les variables métier, encode et sépare le jeu de données pour PyTorch.

    Parameters:
        df (pd.DataFrame): Dataframe brut d'entrée.
        target_type (str): 'binary' (prédit Target) ou 'multiclass' (prédit Failure Type).

    Returns:
        X_train, X_test, y_train, y_test, preprocessor, label_encoder
    """
    data = df.copy()

    # 1. Gestion des fuites de données (Target Leakage) & Suppression des identifiants
    if target_type == "multiclass":
        target_col = "Failure Type"
        drop_cols = ["UDI", "Product ID", "Target"]
    elif target_type == "binary":
        target_col = "Target"
        drop_cols = ["UDI", "Product ID", "Failure Type"]
    else:
        raise ValueError("target_type doit être 'binary' ou 'multiclass'")

    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # 2. Ingénierie des caractéristiques physiques (Feature Engineering)
    data["Temp_Diff"] = (
        data["Process temperature [K]"] - data["Air temperature [K]"]
    )
    data["Power"] = (
        data["Torque [Nm]"] * data["Rotational speed [rpm]"]
    )
    data["Overstrain"] = (
        data["Tool wear [min]"] * data["Torque [Nm]"]
    )

    # 3. Séparation des variables explicatives (X) et de la cible (y)
    X = data.drop(columns=[target_col])
    y_raw = data[target_col]

    # Encodage des classes texte en entiers (0, 1, 2, 3, 4, 5) pour PyTorch
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 4. Configuration des transformateurs de colonnes
    categorical_cols = ["Type"]
    type_order = [["L", "M", "H"]]

    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temp_Diff",
        "Power",
        "Overstrain"
    ]

    cat_transformer = OrdinalEncoder(categories=type_order)
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_cols),
            ("num", num_transformer, numeric_cols)
        ]
    )

    # 5. Division Stratifiée Train/Test (80% / 20%)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 6. Normalisation et encodage
    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    # Reconstruction des DataFrames avec les noms de colonnes
    feature_names = categorical_cols + numeric_cols
    X_train = pd.DataFrame(X_train_array, columns=feature_names)
    X_test = pd.DataFrame(X_test_array, columns=feature_names)

    return X_train, X_test, y_train, y_test, preprocessor, label_encoder


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess_predictive_maintenance(
        df, target_type="multiclass"
    )

    print("--- Prétraitement Multiclasse effectué avec succès ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print("\nCorrespondance des classes (Label Encoder) :")
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"  Classe {idx} -> {class_name}")

    print("\nRépartition des classes dans y_train (effectifs) :")
    print(pd.Series(y_train).value_counts())