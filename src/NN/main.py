from pathlib import Path

import click
import torch
from torch.utils import data
import os
import datahandler
import numpy as np
from model import createDeepLabv3Rn50
from model import createDeepLabv3Rn101
from model import createFCNRn50
from model import createFCNRn101
from model import createUNet
from trainer import train_model


@click.command()
@click.option("--data_directory", required=True, help="Specify the data directory.")
@click.option("--exp_directory", required=True, help="Specify the experiment directory.")
@click.option("--epochs", default=25, type=int,
              help="Specify the number of epochs you want to run the experiment for.")
@click.option("--models", default='DeepLabV3Rn50', type=str, help="Specify the model.")
@click.option("--n_class", required=True, type=int, help="Specify number of class.")
@click.option("--batch_size", default=4, type=int, help="Specify the batch size for the dataloader.")
@click.option('--patience', required=False, type=int, default=15,
              help='Patience before reducing LR (ReduceLROnPlateau)')
@click.option("--train_percent", default=1.0, type=float, help="Specify the percentage of image to train (0.1 - 1)")

def main(data_directory, exp_directory, epochs, batch_size, patience, train_percent, models, n_class):
    # Select models
    if models == 'DeepLabV3Rn50':
        model = createDeepLabv3Rn50(n_class)
    elif models == 'DeepLabV3Rn101':
        model = createDeepLabv3Rn101(n_class)
    elif models == 'FCNetRn50':
        model = createFCNRn50(n_class)
    elif models == 'FCNetRn101':
        model = createFCNRn101(n_class)
    elif models == 'UNet':
        model = createUNet(n_class)
    model.train()
    data_directory = Path(data_directory)

    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Configurate the reduction of learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=patience)

    # Read number of file to train
    num_train = os.listdir(str(data_directory) + '/Train/Images/')

    # Specify parameters to save in .txt
    parameters = [str(models) + 'Parameters' + '\n'
                  '- Optimizer: ' + str(type(optimizer).__name__) + '\n',
                  '- lr: ' + str(optimizer.param_groups[0]['lr']) + '\n',
                  '- Batch Size: ' + str(batch_size) + '\n',
                  '- Epoch: ' + str(epochs)+ '\n',
                  '- Percent of data to train: ' + str(int(train_percent * 100)) + '%' + '\n',
                  '- Number of images to train: ' + str(int(np.ceil(len(num_train)*train_percent))) + '\n']

    with open(str(exp_directory) + '/Parameters.txt', 'w') as f:
        f.writelines(parameters)
        f.close()

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_sep_folder(data_directory, fraction_train=train_percent,
                                                        batch_size=batch_size)

    # Train model
    _ = train_model(model,
                    n_class,
                    dataloaders,
                    optimizer,
                    scheduler,
                    bpath=exp_directory,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    main()
