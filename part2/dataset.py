import torch
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Import du script de prétraitement
from preprocess import preprocess_predictive_maintenance, DATA_PATH


class PredictiveMaintenanceDataset(Dataset):
    """
    Dataset PyTorch personnalisé pour la classification multiclasse de maintenance prédictive.
    """

    def __init__(self, X, y):
        # Conversion des DataFrames/Arrays Pandas/NumPy en Tenseurs PyTorch Float32/Long
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size: int = 64, use_smote: bool = True):
    """
    Charge les données, applique le prétraitement et SMOTE (si activé),
    puis renvoie les DataLoader PyTorch ainsi que les poids des classes.
    """
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess_predictive_maintenance(
        df, target_type="multiclass"
    )

    # --- Étape 3 du TP : Équilibrage par SMOTE sur l'ensemble d'entraînement uniquement ---
    if use_smote:
        # Note : On n'applique JAMAIS SMOTE sur le jeu de test !
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Forme de X_train après SMOTE : {X_train_res.shape}")
        print("Distribution des classes dans y_train après SMOTE :")
        print(pd.Series(y_train_res).value_counts())
    else:
        X_train_res, y_train_res = X_train, y_train

    # Création des Datasets
    train_dataset = PredictiveMaintenanceDataset(X_train_res, y_train_res)
    test_dataset = PredictiveMaintenanceDataset(X_test, y_test)

    # Création des DataLoaders PyTorch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calcul des poids de classe pour CrossEntropyLoss (au cas où SMOTE n'est pas utilisé)
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return train_loader, test_loader, class_weights, label_encoder


if __name__ == "__main__":
    train_loader, test_loader, class_weights, label_encoder = get_dataloaders(batch_size=64, use_smote=True)

    # Vérification d'un batch
    features, labels = next(iter(train_loader))
    print(f"\nTaille d'un batch de caractéristiques : {features.shape}")  # [64, 9]
    print(f"Taille d'un batch de labels :          {labels.shape}")  # [64]