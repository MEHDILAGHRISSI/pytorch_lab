import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from dataset import get_dataloaders
from model import PredictiveMaintenanceDNN
from train import evaluate


def run_regularization_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution de la comparaison de régularisation sur : {device}\n")

    # Hyperparamètres de base
    epochs = 25
    batch_size = 128
    learning_rate = 0.0005

    # Définition des variantes de régularisation
    configs = {
        "1. Aucune régularisation": {
            "use_batch_norm": False,
            "dropout_rate": 0.0,
            "weight_decay": 0.0
        },
        "2. Dropout uniquement (0.3)": {
            "use_batch_norm": False,
            "dropout_rate": 0.3,
            "weight_decay": 0.0
        },
        "3. Weight Decay L2 (1e-3)": {
            "use_batch_norm": False,
            "dropout_rate": 0.0,
            "weight_decay": 0.001
        },
        "4. Combinée (BatchNorm + Dropout + L2)": {
            "use_batch_norm": True,
            "dropout_rate": 0.2,
            "weight_decay": 0.0001
        }
    }

    train_loader, test_loader, _, label_encoder = get_dataloaders(
        batch_size=batch_size,
        use_smote=True
    )

    results = []
    histories = {}

    for name, cfg in configs.items():
        print(f"=== Entraînement : {name} ===")

        # Instanciation du modèle avec la configuration spécifique
        model = PredictiveMaintenanceDNN(
            dropout_rate=cfg["dropout_rate"],
            use_batch_norm=cfg["use_batch_norm"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=cfg["weight_decay"]
        )

        history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total

            test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

        histories[name] = history

        # Évaluation finale des métriques
        _, _, y_train_true, y_train_pred = evaluate(model, train_loader, criterion, device)
        _, _, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, device)

        train_acc_final = accuracy_score(y_train_true, y_train_pred)
        test_acc_final = accuracy_score(y_test_true, y_test_pred)
        macro_f1_test = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
        weighted_f1_test = f1_score(y_test_true, y_test_pred, average="weighted", zero_division=0)

        # Calcul du surapprentissage (Gap entre Train et Test)
        overfitting_gap = train_acc_final - test_acc_final

        print(f"--> Train Acc: {train_acc_final * 100:.2f}% | Test Acc: {test_acc_final * 100:.2f}% | "
              f"Gap: {overfitting_gap * 100:.2f}% | Macro F1: {macro_f1_test:.4f}\n")

        results.append({
            "Configuration": name,
            "Train Acc (%)": round(train_acc_final * 100, 2),
            "Test Acc (%)": round(test_acc_final * 100, 2),
            "Gap Overfitting (%)": round(overfitting_gap * 100, 2),
            "Test Loss": round(history["test_loss"][-1], 4),
            "Macro F1": round(macro_f1_test, 4),
            "Weighted F1": round(weighted_f1_test, 4)
        })

    # Tableau comparatif final
    df_compare = pd.DataFrame(results)
    print("=" * 80)
    print("TABLEAU COMPARATIF DES TECHNIQUES DE RÉGULARISATION")
    print("=" * 80)
    print(df_compare.to_string(index=False))
    df_compare.to_csv("regularization_comparison.csv", index=False)

    # Courbes comparatives
    plot_regularization_comparison(histories, epochs)


def plot_regularization_comparison(histories, epochs):
    """
    Trace les pertes de test et exactitudes de test pour chaque variante.
    """
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ["red", "orange", "purple", "green"]

    for (name, hist), color in zip(histories.items(), colors):
        ax1.plot(epochs_range, hist["test_loss"], label=name, color=color, linewidth=2)
        ax2.plot(epochs_range, hist["test_acc"], label=name, color=color, linewidth=2)

    ax1.set_title("Test Loss vs Epochs (Effet de la Régularisation)")
    ax1.set_xlabel("Époques")
    ax1.set_ylabel("Test Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Test Accuracy vs Epochs (Effet de la Régularisation)")
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Test Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("regularization_comparison.png", dpi=300)
    print("\nGraphique comparatif sauvegardé sous 'regularization_comparison.png'.")
    plt.show()


if __name__ == "__main__":
    run_regularization_experiment()