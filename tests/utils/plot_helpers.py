import matplotlib.pyplot as plt


def save_dewarp_comparison_figure(
    grad_left, grad_right, left_data, right_data, output_path
):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(grad_left, cmap="gray")
    axs[0, 0].set_title("Original Left")
    axs[0, 1].imshow(grad_right, cmap="gray")
    axs[0, 1].set_title("Original Right")

    axs[1, 0].imshow(left_data, cmap="gray")
    axs[1, 0].set_title("Dewarped Left")
    axs[1, 1].imshow(right_data, cmap="gray")
    axs[1, 1].set_title("Dewarped Right")

    for ax in axs.flatten():
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)
