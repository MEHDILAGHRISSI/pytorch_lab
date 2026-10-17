import matplotlib.pyplot as plt


def draw_neural_network(layer_sizes, filename="dnn_architecture.jpg"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    left, right, bottom, top = 0.08, 0.92, 0.12, 0.78
    layer_names = [
        "Input Layer\n(10 Features)",
        "Hidden 1\n(64 Neurons)",
        "Hidden 2\n(32 Neurons)",
        "Hidden 3\n(16 Neurons)",
        "Output Layer\n(1 Target Return)"
    ]

    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    for i, n_neurons in enumerate(layer_sizes):
        display_n = min(n_neurons, 8)
        x = left + i * h_spacing

        # Center nodes vertically per layer
        y_positions = [0.45 + (j - (display_n - 1) / 2.0) * 0.08 for j in range(display_n)]

        # Connections to next layer
        if i < len(layer_sizes) - 1:
            next_n = min(layer_sizes[i + 1], 8)
            next_y_positions = [0.45 + (k - (next_n - 1) / 2.0) * 0.08 for k in range(next_n)]
            next_x = left + (i + 1) * h_spacing

            for y1 in y_positions:
                for y2 in next_y_positions:
                    ax.plot([x, next_x], [y1, y2], color='#b0b0b0', alpha=0.35, lw=0.7, zorder=1)

        # Plot sharp node circles
        ax.scatter([x] * display_n, y_positions, s=220, color='#1f77b4', edgecolors='black', linewidths=1.2, zorder=4)

        # Layer Labels
        ax.text(x, 0.78, layer_names[i], fontsize=10, ha='center', va='bottom', weight='bold')

    plt.suptitle("PyTorch Stock Return Regression DNN Architecture", fontsize=14, weight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Updated diagram saved to '{filename}'")


draw_neural_network([10, 64, 32, 16, 1])