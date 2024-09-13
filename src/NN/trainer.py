import copy
import csv
import os
import time

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchmetrics import JaccardIndex
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


EPS = 1e-6

def get_weight(file_name, num_class):
    num_classes = num_class + 1
    num_pixels_per_class = [0] * num_classes
    for sample in iter(file_name):
        labels = sample['target']
        for c in range(1, num_classes):
            num_pixels_per_class[c] += torch.sum(labels == c).item()

    # Calculate max number of pixel per image
    total_num_pixels = sum(num_pixels_per_class)

    # Calculate percentage of pixel per image in function of total number of pixels
    pixel_percentages = [num_pixels / total_num_pixels for num_pixels in num_pixels_per_class]

    pixel_percentages.remove(0)
    print(pixel_percentages)
    return pixel_percentages

def ponderated_masked_mse(pred, labels, mask, class_count):
    mask = mask/mask
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (pred - labels) ** 2
    loss /= class_count
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def train_model(model, num_class, dataloaders, optimizer, scheduler, bpath, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set Jaccad index (IoU)
    jaccard = JaccardIndex(task="multiclass", num_classes=num_class, ignore_index=0, average='micro').to(device)

    # Get percentage of labeled per class
    class_count = get_weight(dataloaders['Train'], num_class)
    class_count_expanded = torch.tensor(class_count).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).permute(1, 0, 2, 3).to(device)

    model.to(device)
    # Initialize the log file for training and validating loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Val_loss', 'IoU', 'lr']
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Set early stop
    early_stopper = EarlyStopper(patience=3, min_delta=5)

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        batchsummary['lr'] = optimizer.param_groups[0]['lr']

        for phase in ['Train', 'Val']:
            if phase == 'Train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                target = sample['target'].to(device)

                # One hot encoding
                target_ohe = F.one_hot(target.to(torch.int64), num_classes=num_class+1).squeeze(1).permute(0, 3, 1, 2).float()
                target_ohe = target_ohe[:, 1:num_class+1, :, :]

                # zero the parameter gradients
                optimizer.zero_grad()

                # Training parameters
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)['out']

                    loss = ponderated_masked_mse(outputs, target_ohe, target, class_count_expanded)
                    
                    batchsummary['IoU'] = jaccard(outputs, target_ohe.int()).item()

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                    if phase == 'Val':
                        loss_validation = loss

            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))

        for field in fieldnames[4:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)

        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Val' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

        # Reduce LR on Plateau after patience reached
        prevLR = optimizer.param_groups[0]['lr']
        scheduler.step(loss_validation)
        currLR = optimizer.param_groups[0]['lr']
        if currLR is not prevLR and scheduler.num_bad_epochs == 0:
            print("Plateau Reached!")

        if prevLR < 2 * scheduler.eps and scheduler.num_bad_epochs >= scheduler.patience:
            print("Plateau Reached and no more reduction -> Exiting Loop")
            parameters = ['- Exiting at Epoch by lr (Plateau Reached): ' + str(epoch) + '\n']

            with open(str(bpath) + '/Parameters.txt', 'a') as f:
                f.writelines(parameters)
                f.close()

            break

        # Exit if Early Stopping
        if early_stopper.early_stop(epoch_loss.item()):
            print('Early Stop')
            parameters = ['- Exiting at Epoch by Early Stopping: ' + str(epoch) + '\n']

            with open(str(bpath) + '/Parameters.txt', 'a') as f:
                f.writelines(parameters)
                f.close()

            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model
