import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from dataset import get_dataloaders
from model import PredictiveMaintenanceDNN


def evaluate(model, dataloader, criterion, device):
    """
    Évalue le modèle sur un DataLoader donné et retourne la perte moyenne,
    l'exactitude, ainsi que les vraies étiquettes et prédites.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_targets), np.array(all_preds)


def train_model(
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        use_smote: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}\n")

    # 1. Chargement des données
    train_loader, test_loader, class_weights, label_encoder = get_dataloaders(
        batch_size=batch_size,
        use_smote=use_smote
    )

    # 2. Initialisation du modèle, perte et optimiseur
    model = PredictiveMaintenanceDNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Historiques pour les graphiques
    history = {
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": []
    }

    # 3. Boucle d'entraînement
    print("--- Début de l'entraînement ---")
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

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Évaluation sur le jeu de test
        epoch_test_loss, epoch_test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(epoch_train_loss)
        history["test_loss"].append(epoch_test_loss)
        history["train_acc"].append(epoch_train_acc)
        history["test_acc"].append(epoch_test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Époque [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc * 100:.2f}% | "
                f"Test Loss: {epoch_test_loss:.4f} - Test Acc: {epoch_test_acc * 100:.2f}%"
            )

    # 4. Évaluation finale détaillée (Étape 7 du TP)
    _, _, y_train_true, y_train_pred = evaluate(model, train_loader, criterion, device)
    _, _, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 50)
    print("RÉSULTATS ET MÉTRIQUES D'ÉVALUATION (TEST SET)")
    print("=" * 50)
    print(classification_report(
        y_test_true, y_test_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    ))

    # 5. Visualisation des courbes (Étape 6 du TP)
    plot_metrics(history, epochs)

    return model, history, label_encoder


def plot_metrics(history, epochs):
    """
    Génère et sauvegarde les deux graphiques : Loss/Epochs et Accuracy/Epochs.
    """
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique Loss
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    ax1.plot(epochs_range, history["test_loss"], label="Test Loss", color="red", linestyle="--", linewidth=2)
    ax1.set_title("Loss vs Epochs")
    ax1.set_xlabel("Époques")
    ax1.set_ylabel("Perte (CrossEntropy)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Graphique Accuracy
    ax2.plot(epochs_range, history["train_acc"], label="Train Accuracy", color="blue", linewidth=2)
    ax2.plot(epochs_range, history["test_acc"], label="Test Accuracy", color="red", linestyle="--", linewidth=2)
    ax2.set_title("Accuracy vs Epochs")
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Exactitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_performance.png", dpi=300)
    print("\nGraphiques sauvegardés sous le nom 'training_performance.png'.")
    plt.show()


if __name__ == "__main__":
    train_model(epochs=30, batch_size=64, learning_rate=0.001)