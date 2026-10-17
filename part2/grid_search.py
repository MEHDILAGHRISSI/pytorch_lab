import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import pandas as pd
from dataset import get_dataloaders
from model import PredictiveMaintenanceDNN
from train import evaluate


def run_grid_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution du GridSearch sur : {device}\n")

    # Grille d'hyperparamètres à tester
    param_grid = {
        "learning_rate": [1e-3, 5e-4],
        "batch_size": [64, 128],
        "dropout_rate": [0.2, 0.4],
        "weight_decay": [1e-4, 1e-3],
        "epochs": [20]
    }

    keys, values = zip(*param_grid.items())
    permutations = [dict(zip(keys, v)) for v in product(*values)]

    results = []

    print(f"Nombre total de combinaisons à tester : {len(permutations)}\n")

    for idx, params in enumerate(permutations, 1):
        print(f"--- Configuration [{idx}/{len(permutations)}] : {params} ---")

        train_loader, test_loader, _, label_encoder = get_dataloaders(
            batch_size=params["batch_size"],
            use_smote=True
        )

        model = PredictiveMaintenanceDNN(
            dropout_rate=params["dropout_rate"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
        )

        # Entraînement rapide pour la grille
        for epoch in range(params["epochs"]):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Évaluation
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

        # Calcul du F1 Macro pour ne pas privilégier uniquement la classe majoritaire
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"--> Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f} | Macro F1: {macro_f1:.4f}\n")

        results.append({
            **params,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "macro_f1": macro_f1
        })

    # Conversion des résultats en DataFrame
    df_results = pd.DataFrame(results).sort_values(by="macro_f1", ascending=False)
    print("=" * 60)
    print("MEILLEURES CONFIGURATIONS (Classement par Macro F1) :")
    print("=" * 60)
    print(df_results.to_string(index=False))

    df_results.to_csv("grid_search_results.csv", index=False)
    print("\nRésultats enregistrés dans 'grid_search_results.csv'.")


if __name__ == "__main__":
    run_grid_search()