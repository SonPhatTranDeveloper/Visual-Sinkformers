import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load the cv result
    transformers_result = np.load("cv_softmax.npy")
    sinkformers_result = np.load("cv_sinkhorn.npy")

    # Extract training and testing
    trans_train_loss_array, trans_val_loss_array, trans_train_accuracy_array, trans_val_accuracy_array = (
        transformers_result
    )

    sink_train_loss_array, sink_val_loss_array, sink_train_accuracy_array, sink_val_accuracy_array = (
        sinkformers_result
    )

    # Plot the result
    epochs = [i for i in range(1, 251)]
    fig, axs = plt.subplots(1, 1, figsize=(6, 10))
    axs.plot(epochs, trans_train_accuracy_array[50:], color="blue", linewidth=3, alpha=0.5, label="Train")
    axs.plot(epochs, trans_val_accuracy_array[50:], color="blue", linewidth=3, alpha=0.5, label="Validation")
    axs.plot(epochs, sink_train_accuracy_array[50:], color="red", linewidth=3, alpha=0.5, label="Train")
    axs.plot(epochs, sink_val_accuracy_array[50:], color="red", linewidth=3, alpha=0.5, label="Validation")
    axs.set_title("Training and validation accuracies")

    # Display the plot
    plt.legend()
    plt.show()
    fig.savefig("Dogs and Cats Dataset Result.png")