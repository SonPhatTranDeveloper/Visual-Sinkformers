"""
Define a trainer for the Transformer encoder block
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import time

from components.nlp import NLPTransformerEncoder


class SentimentAnalysisTrainer:
    def __init__(self, args, train_dataloader, test_dataloader, tokenizer, n_iter, mode="softmax"):
        """
        Create a trainer for Sentiment Analysis
        :param args: the various argument parsed
        :param train_dataloader: the data loader for training data
        :param test_dataloader: the data loader for testing data
        :param tokenizer: the tokenizer
        :param n_iter: number of sinkhorn iteration if sinkhorn is used
        :param mode: softmax or sinkhorn
        """
        # Cache the components
        self.args = args
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_token_id

        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create NLP transformer encoder
        # based on the mode
        model = NLPTransformerEncoder(
            vocab_size=self.vocab_size,
            sequence_length=args.max_sequence_length,
            pad_id=self.pad_id,
            n_layers=args.n_layers,
            d_model=args.hidden,
            n_heads=args.n_attn_heads,
            p_dropout=args.dropout,
            d_hidden=args.ffn_hidden,
            mode=mode,
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
        Train the model for one epoch
        :param epoch: the current epoch
        :return: average loss per batch and training accuracy
        """
        # Get the current time
        current_time = time.time()

        # Reduce the learning rate by 10 if we reach epoch 12
        if epoch == 12:
            for g in self.optimizer.param_groups:
                g['lr'] /= 10

        # Initialize the losses and accuracy
        losses, accs = 0, 0

        # Get the number of batches and the number of samples
        n_batches, n_samples = len(self.train_dataloader), len(self.train_dataloader.dataset)

        # Put the model into train mode
        self.model.train()

        # Cache the attention weights
        for i, batch in enumerate(self.train_dataloader):
            # print(f"Batch {i}")

            # Map to device
            # inputs have size (batch_size, max_sequence_length)
            # labels have size (batch_size, )
            inputs, labels = map(lambda x: x.to(self.device), batch)

            # Perform the forward pass
            # attention_weights is a list containing n_layers of
            # tensor of size (batch_size, n_heads, sequence_length, sequence_length)
            outputs, attention_weights = self.model(inputs)

            # Calculate the current loss and accuracy
            loss = self.criterion(outputs, labels)
            losses += loss.item()
            acc = (outputs.argmax(dim=-1) == labels).sum()
            accs += acc.item()

            # Perform the backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Display the current batch accuracy after 5 epochs
            if i % (n_batches // 5) == 0 and i != 0:
                print('Iteration {} ({}/{})\tLoss: {:.4f} Acc: {:4f}%'.format(
                    i, i, n_batches, losses / i, accs / (i * self.args.batch_size) * 100.))

        # Calculate the training time
        print(time.time() - current_time)

        # Calculate average loss per batch and accuracy
        losses_b = losses / n_batches
        acc_ns = accs / n_samples * 100.

        # Display the current status
        print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, losses_b, acc_ns))

        # Return the current stats
        return losses_b, acc_ns

    def validate(self, epoch):
        """
        Validate the model using the validation dataset
        :param epoch: current epoch
        :return: loss per batch and accuracy
        """
        # Initialize the losses and accuracy
        losses, accs = 0, 0

        # Get the number of batches and the number of samples of the test loader
        n_batches, n_samples = len(self.test_dataloader), len(self.test_dataloader.dataset)

        # Put the model into eval mode
        self.model.eval()

        # Validate
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                # Map to device
                # inputs have size (batch_size, max_sequence_length)
                # labels have size (batch_size, )
                inputs, labels = map(lambda x: x.to(self.device), batch)

                # Perform the forward pass
                # attention_weights is a list containing n_layers of
                # tensor of size (batch_size, n_heads, sequence_length, sequence_length)
                outputs, attention_weights = self.model(inputs)

                # Calculate the current loss and accuracy
                loss = self.criterion(outputs, labels)
                losses += loss.item()
                acc = (outputs.argmax(dim=-1) == labels).sum()
                accs += acc.item()

        # Calculate average loss per batch and accuracy
        losses_b = losses / n_batches
        acc_ns = accs / n_samples * 100.

        # Display the current stats
        print('Train Epoch: {}\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(epoch, losses_b, acc_ns))

        # Return the current stats
        return losses_b, acc_ns

    def save(self, epoch, model_prefix='model', root='.model'):
        """
        Save the current transformer model
        :param epoch: the current training epoch
        :param model_prefix: the model name
        :param root: the model file name
        :return: None
        """
        # Fine the path to save the model
        path = Path(root) / (model_prefix + '.ep%d' % epoch)

        # Create the model folder if not exist
        if not path.parent.exists():
            path.parent.mkdir()

        # Save the model
        torch.save(self.model, path)