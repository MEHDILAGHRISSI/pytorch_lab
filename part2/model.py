import torch
import torch.nn as nn


class PredictiveMaintenanceDNN(nn.Module):
    """
    Réseau de neurones dense (MLP) pour la classification multiclasse de maintenance prédictive.

    Args:
        input_dim (int): Nombre de caractéristiques en entrée (par défaut: 9).
        num_classes (int): Nombre de classes en sortie (par défaut: 6).
        hidden_units (list): Nombre de neurones par couche cachée.
        dropout_rate (float): Taux de Dropout pour la régularisation.
        use_batch_norm (bool): Activer ou non BatchNorm1d.
    """

    def __init__(
            self,
            input_dim: int = 9,
            num_classes: int = 6,
            hidden_units: list = [128, 64, 32],
            dropout_rate: float = 0.2,
            use_batch_norm: bool = True
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Construction dynamique des couches cachées
        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = units

        # Couche de sortie : 6 logits pour nn.CrossEntropyLoss
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    # Test de validation du modèle avec un tenseur factice
    model = PredictiveMaintenanceDNN()
    dummy_batch = torch.randn(64, 9)
    output = model(dummy_batch)

    print("--- Architecture du Modèle ---")
    print(model)
    print(f"\nForme de l'entrée : {dummy_batch.shape}")
    print(f"Forme de la sortie : {output.shape}  (attendu : [64, 6])")