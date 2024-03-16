"""
This file defines the trainer for the cats and dogs dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim

import time

from components.cv import VisualTransformerClassification


class CatsAndDogsTrainer:
    def __init__(self, args, train_dataloader, test_dataloader, n_iter, mode="softmax"):
        """
        Create a trainer for Cats and Dogs classification
        :param args: the various argument parsed
        :param train_dataloader: the data loader for training data
        :param test_dataloader: the data loader for testing data
        :param n_iter: number of sinkhorn iteration if sinkhorn is used
        :param mode: softmax or sinkhorn
        """
        # Cache the parameters
        self.args = args
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # Define the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create the model
        model = VisualTransformerClassification(
            image_width=args.image_width, image_height=args.image_height,
            patch_width=args.patch_width, patch_height=args.patch_height,
            n_classes=args.n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            pooling=args.pooling,
            n_channels=args.n_channels,
            d_head=args.d_head,
            p_dropout=args.p_dropout,
            p_emb_dropout=args.p_emb_dropout,
            attention_class=mode,
            n_iter=n_iter
        )

        # Reset the parameters
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Set parallel training data
        model = nn.DataParallel(model)

        # Set the model
        self.model = model
        self.model = self.model.to(self.device)

        # Create optimizer and cross-entropy loss function
        self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch):
        """
        Train the Visual Transformer for one epoch
        :param epoch: the current epoch
        :return: epoch loss and accuracy
        """
        # Get the current time
        current_time = time.time()

        # Adjust the learning rate on the 250 epoch
        if epoch == 250:
            for g in self.optimizer.param_groups:
                g['lr'] /= 10

        # Get the number of batches and the number of samples of the test loader
        n_batches, n_samples = len(self.train_dataloader), len(self.train_dataloader.dataset)

        # Initialize the loss and accuracy
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        # Put the model into train mode
        self.model.train()

        # Calculate the loss and accuracy
        for image, label in self.train_dataloader:
            # print("PROCESSING")
            # Map image and label to device
            image = image.to(self.device)
            label = label.to(self.device)

            # Forward pass through visual transformer
            output, attn_weights = self.model(image)
            loss = self.criterion(output, label)

            # Backward pass through visual transformer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate the loss and accuracy
            acc = (output.argmax(dim=1) == label).float().sum()
            epoch_accuracy += acc.item()
            epoch_loss += loss.item()

        # Calculate the loss and accuracy
        epoch_loss = epoch_loss / n_batches
        epoch_accuracy = epoch_accuracy / n_samples * 100

        # Calculate the training time
        print(time.time() - current_time)

        # Display the current status
        print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, epoch_loss, epoch_accuracy))

        return epoch_accuracy, epoch_accuracy

    def validate(self, epoch):
        """
        Perform the validation at epoch
        :param epoch: the current epoch
        :return: the epoch loss and accuracy
        """
        # Get the number of batches and the number of samples of the test loader
        n_batches, n_samples = len(self.test_dataloader), len(self.test_dataloader.dataset)

        # Put the model into eval mode
        self.model.eval()

        # Validate
        with torch.no_grad():
            epoch_val_accuracy = 0.0
            epoch_val_loss = 0.0

            for data, label in self.test_dataloader:
                # Map image and label to device
                data = data.to(self.device)
                label = label.to(self.device)

                # Forward pass through the Visual Transformer
                val_output, val_attn_weights = self.model(data)
                val_loss = self.criterion(val_output, label)

                # Get the loss and accuracy
                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc.item()
                epoch_val_loss += val_loss.item()

        # Calculate the validation accuracy and loss
        epoch_val_loss = epoch_val_loss / n_batches
        epoch_val_accuracy = epoch_val_accuracy / n_samples * 100

        # Display the current stats
        print('Validation Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, epoch_val_loss,
                                                                            epoch_val_accuracy))
        return epoch_val_loss, epoch_val_accuracy

    def save(self, model_path, epoch):
        """
        Save the current model
        :param model_path: the saved model path
        :param epoch: the current epoch
        :return: None
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
