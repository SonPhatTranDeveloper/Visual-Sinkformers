import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the numpy result
    transformers_result = np.load("transformers_nlp_sentiment_analysis.npy")
    sinkformers_result = np.load("sinkformers_nlp_sentiment_analysis.npy")

    # Get the individual result
    trans_train_loss_array, trans_val_loss_array, trans_train_accuracy_array, trans_val_accuracy_array = (
        transformers_result
    )
    sink_train_loss_array, sink_val_loss_array, sink_train_accuracy_array, sink_val_accuracy_array = (
        sinkformers_result
    )

    # Plot the result
    epochs = [i for i in range(1, 16)]
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    axs[0].plot(epochs, trans_train_loss_array, color="blue", linewidth=3, alpha=0.5, label="Transformers")
    axs[0].plot(epochs, sink_train_loss_array, color="orange", linewidth=3, alpha=0.5, label="Sinkformers")
    axs[0].set_title("Train costs")

    axs[1].plot(epochs, trans_val_loss_array, color="blue", linewidth=3, alpha=0.5, label="Transformers")
    axs[1].plot(epochs, sink_val_loss_array, color="orange", linewidth=3, alpha=0.5, label="Sinkformers")
    axs[1].set_title("Validation costs")

    axs[2].plot(epochs, trans_train_accuracy_array, color="blue", linewidth=3, alpha=0.5, label="Transformers")
    axs[2].plot(epochs, sink_train_accuracy_array, color="orange", linewidth=3, alpha=0.5, label="Sinkformers")
    axs[2].set_title("Train accuracies")

    axs[3].plot(epochs, trans_val_accuracy_array, color="blue", linewidth=3, alpha=0.5, label="Transformers")
    axs[3].plot(epochs, sink_val_accuracy_array, color="orange", linewidth=3, alpha=0.5, label="Sinkformers")
    axs[3].set_title("Validation accuracies")

    # Show the plot
    plt.legend()
    plt.show()
    fig.savefig("NLP Sentiment Analysis Result.png")


