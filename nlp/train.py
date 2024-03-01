"""
This file contains the code to train the transformer model on sentiment analysis
Run this file to train the Transformers on the sentiment analysis task
Author: Son Phat Tran
"""
from mosestokenizer import MosesTokenizer

import torch
import torch.utils.data
from torch.utils.data import DataLoader

import argparse

import numpy as np

from tokenizer import Tokenizer
from data_utils import create_dataset
from trainer import SentimentAnalysisTrainer
from imdb_dataset import IMDBDataset


def train(arguments):
    """
    Train and save the NLP transformer model on sentiment analysis task
    :param arguments: all the arguments required to train the model
    :return:
    """
    # Save address of the losses
    stats_save_dir = arguments.save_dir
    n_it = arguments.n_iter
    seed = arguments.seed
    stats_save_address = stats_save_dir + '/%d_it_%d.npy' % (n_it, seed)

    # Create the tokenizer
    tokenizer = Tokenizer(
        tokenizer=MosesTokenizer(),
        vocab_file_path="nlp/imdb.vocab"
    )

    # Create data loader
    imdb = IMDBDataset("data/imdb_dataset.csv")
    dataset = create_dataset(
        sentiment_analysis_dataset=imdb,
        tokenizer=tokenizer
    )

    # Split the dataset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [40000, 10000])

    # Display info
    print('Train dataset of size %d' % len(train_dataset))
    print('Test dataset of size %d' % len(test_dataset))

    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=arguments.batch_size, shuffle=True)

    # Build the trainer
    trainer = SentimentAnalysisTrainer(
        args=arguments,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        tokenizer=tokenizer,
        n_iter=arguments.n_iter,
        mode=arguments.mode
    )

    # Start training, validation and save loop
    val_loss_array = []
    train_loss_array = []
    val_accuracy_array = []
    train_accuracy_array = []

    # Train & Validate
    for epoch in range(1, arguments.epochs + 1):
        # Train the model for thi epoch
        epoch_loss, epoch_accuracy = trainer.train(epoch)

        # Validate the model
        epoch_val_loss, epoch_val_accuracy = trainer.validate(epoch)

        # Save the model
        trainer.save(epoch, arguments.output_model_prefix)

        # Save the training and validation accuracy
        val_accuracy_array.append(epoch_val_accuracy)
        train_accuracy_array.append(epoch_accuracy)

        # Save the validation and training loss
        val_loss_array.append(epoch_val_loss)
        train_loss_array.append(epoch_loss)

        # Save the model
        trainer.save(epoch, arguments.output_model_prefix)

        # Save the training and validation result
        losses = np.asarray([train_loss_array, val_loss_array, train_accuracy_array, val_accuracy_array])
        np.save(stats_save_address, losses)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()

    # House-keeping
    parser.add_argument('--dataset', default='imdb', type=str, help='dataset')
    parser.add_argument('--vocab_file', default='wiki.vocab', type=str, help='vocabulary path')
    parser.add_argument('--output_model_prefix', default='model', type=str, help='output model name prefix')

    # Input parameters
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--max_sequence_length', default=512, type=int, help='the maximum size of the input sequence')

    # Softmax or sinkhorn
    parser.add_argument("--mode", default="sinkhorn", help="use softmax or sinkhorn normalization")

    # Train parameters
    parser.add_argument('--epochs', default=15, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--hidden', default=256, type=int, help='the number of expected features in the transformer')
    parser.add_argument('--n_layers', default=6, type=int,
                        help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_attn_heads', default=8, type=int, help='the number of multi-head attention heads')
    parser.add_argument('--dropout', default=0.1, type=float, help='the residual dropout value')
    parser.add_argument('--ffn_hidden', default=1024, type=int, help='the dimension of the feedforward network')
    parser.add_argument('--save_dir', default='results', type=str, help='save dir')
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    # Parse the argument
    args = parser.parse_args()

    # Initialize the training
    train(args)

