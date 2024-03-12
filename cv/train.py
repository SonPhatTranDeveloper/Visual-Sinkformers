"""
This file defines the train script for the Visual Transformers model
"""
import torch
import torch.utils.data
from torch.utils.data import DataLoader

import numpy as np

import argparse

from trainer import CatsAndDogsTrainer

from data_utils import create_file_lists
from cats_and_dogs_dataset import create_image_transformation, CatsAndDogsDataset


def train(arguments):
    """
    Train and save the Visual Transformer model
    :param arguments: the parsed arguments
    :return: None
    """
    # Save address of the losses
    stats_save_dir = arguments.save_dir
    n_it = arguments.n_iter
    seed = arguments.seed
    stats_save_address = stats_save_dir + '/%d_it_%d.npy' % (n_it, seed)

    # Create dataloaders
    train_list, validation_list, test_list = create_file_lists(
        train_dir="data/cats-and-dogs/train",
        test_dir="data/cats-and-dogs/test",
        validation_size=0.2,
        random_seed=42
    )

    # Create transformation for train, validation, and test dataset
    TRAIN_TRANSFORMATION = create_image_transformation()
    VALIDATION_TRANSFORMATION = create_image_transformation()

    # Create dataset
    train_dataset = CatsAndDogsDataset(train_list, TRAIN_TRANSFORMATION)
    validation_dataset = CatsAndDogsDataset(validation_list, VALIDATION_TRANSFORMATION)

    # Display info
    print('Train dataset of size %d' % len(train_dataset))
    print('Validation dataset of size %d' % len(validation_dataset))

    # Create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=arguments.batch_size, shuffle=True)

    # Create trainer
    trainer = CatsAndDogsTrainer(
        arguments,
        train_dataloader,
        validation_dataloader,
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
        trainer.save(arguments.output_model_prefix, epoch)

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

    # Output model prefix
    parser.add_argument('--output_model_prefix', default='model.pth', type=str, help='output model name prefix')

    # Image width and height
    parser.add_argument('--image_width', default=224, type=int, help='image width')
    parser.add_argument('--image_height', default=224, type=int, help='image height')
    parser.add_argument('--n_channels', default=3, type=int, help='image channel')
    parser.add_argument('--patch_width', default=16, type=int, help='patch width')
    parser.add_argument('--patch_height', default=16, type=int, help='patch height')

    # Input parameters
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    # Softmax or sinkhorn
    parser.add_argument("--mode", default="softmax", help="use softmax or sinkhorn normalization")
    parser.add_argument("--n_iter", default=1, type=int, help="the number sinkhorn iteration")

    # Train parameters
    parser.add_argument('--epochs', default=300, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--no_cuda', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--n_classes', default=2, type=int, help='the number of class in Visual Transformer')
    parser.add_argument('--d_model', default=128, type=int, help='the number of class in Visual Transformer')
    parser.add_argument('--n_layers', default=6, type=int, help='the number of multi-head self-attention '
                                                                'layers in Visual Transformer')
    parser.add_argument('--n_heads', default=8, type=int, help='the number of heads in multi-head '
                                                               'self-attention in Visual Transformer')
    parser.add_argument('--d_head', default=64, type=int, help='the size of each head in multi-headed '
                                                               'self-attention')
    parser.add_argument('--d_ff', default=128, type=int, help='the hidden size of the feed forward network')
    parser.add_argument('--pooling', default='cls', type=str, help='the type of pooling for Visual Transformer')
    parser.add_argument('--p_dropout', default=0.0, type=int, help='the dropout rate for Transformer Encoder')
    parser.add_argument('--p_emb_dropout', default=0.0, type=int, help='the dropout rate for the positional'
                                                                       'embedding layer')

    # Save parameters
    parser.add_argument('--save_dir', default='results', type=str, help='save dir')
    parser.add_argument("--seed", type=int, default=0)

    # Parse the argument
    args = parser.parse_args()

    # Train the model
    train(args)
