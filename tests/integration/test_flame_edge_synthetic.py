import matplotlib.pyplot as plt
import numpy as np

from flametrack.analysis.flamespread import (
    calculate_edge_data,
    left_most_point_over_threshold,
)


def shifted_flammenkante(x, y, shift_x):
    return y - (1.2 * (x - shift_x) - 20)


def true_flame_x(y_val, shift):
    return (y_val + 20) / 1.2 + shift


def generate_synthetic_stack(n_frames=9, shape=(300, 300), vmin=50, vmax=500):
    """Erzeuge Temperaturdaten + erwartete Flammenkantenpositionen (in Pixeln) für mittlere Bildhöhe."""
    x = np.linspace(0, 100, shape[1])  # Weltkoordinaten x: 0–100
    y = np.linspace(0, 100, shape[0])  # Weltkoordinaten y: 0–100
    X, Y = np.meshgrid(x, y)

    x_shifts = np.linspace(100, -100, n_frames)  # Flamme wandert von rechts nach links
    frames = []
    ground_truth = []

    middle_row = shape[0] // 2
    y_val = y[middle_row]  # Weltkoordinate der mittleren Bildzeile

    for shift in x_shifts:
        flk = shifted_flammenkante(X, Y, shift)
        temp = np.where(flk < 0, vmax, vmin)
        frames.append(temp)

        # Berechne x-Position der Flammenkante bei y_val (in Weltkoordinaten)
        x_phys = true_flame_x(y_val, shift)

        # Begrenze auf sichtbaren Bereich und rechne in Pixelkoordinaten um
        x_phys_clipped = np.clip(x_phys, 0, 100)
        x_pixel = int(np.round(x_phys_clipped / 100 * (shape[1] - 1)))
        ground_truth.append(x_pixel)

    stack = np.stack(frames, axis=-1)  # shape: (H, W, F)
    return stack, ground_truth, x, y, x_shifts


def test_flame_edge_tracking_on_synthetic_data(save_test_plot):
    stack, expected_edges, x_vals, y_vals, x_shifts = generate_synthetic_stack()

    results = calculate_edge_data(
        data=stack,
        find_edge_point=lambda x, params=None: left_most_point_over_threshold(
            x, threshold=250
        ),
        custom_filter=lambda x: x,
    )

    middle_row = stack.shape[0] // 2
    detected_edges = [frame[middle_row] for frame in results]
    error = np.abs(np.array(detected_edges) - np.array(expected_edges))

    # print("Expected edges:", expected_edges)
    # print("Detected edges:", detected_edges)

    # === Plot using contourf and neon green overlay ===
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()
    vmin, vmax = 50, 500

    for i in range(stack.shape[-1]):
        ax = axes[i]

        x = np.linspace(0, 100, stack.shape[1])
        y = np.linspace(0, 100, stack.shape[0])
        X, Y = np.meshgrid(x, y)
        flk = shifted_flammenkante(X, Y, x_shifts[i])
        temp = np.where(flk < 0, vmax, vmin)

        # Plot temperature field with contourf (original style)
        c = ax.contourf(X, Y, temp, levels=100, cmap="jet", vmin=vmin, vmax=vmax)

        # Plot expected flame front as line in world coords
        x_line = np.linspace(0, 100, 300)
        y_line = 1.2 * (x_line - x_shifts[i]) - 20
        ax.plot(
            x_line, y_line, color="red", linestyle="-", linewidth=1, label="Expected"
        )

        # Plot detected edge (pixel coords)
        y_pixel = y
        x_pixel = np.array(results[i])
        x_world = x_pixel / (stack.shape[1] - 1) * 100
        ax.plot(x_world, y_pixel, color="#39FF14", linewidth=3, label="Detected")

        ax.set_title(f"Frame {i + 1}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(c, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("Temperatur")
    plt.suptitle("Detected vs Expected Flame Edges (synthetic test)", fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    save_test_plot(fig, suffix="flame_edge")

    # Final assertion
    assert np.all(error <= 2), f"Flame edge tracking error too large: {error}"
