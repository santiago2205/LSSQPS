import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
import click
import unet
from PIL import Image, ImageFilter
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn import functional as F
from torchmetrics import JaccardIndex

@click.command()
@click.option("--data_directory", required=True, help="Specify the data directory.")
@click.option("--exp_directory", required=True, help="Specify the experiment directory.")


def main(data_directory, exp_directory):
    # Path to folder train output
    path_train = exp_directory

    # Path to Test folder
    path_test = data_directory + '/Test/'

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = torch.load(path_train + '/weights.pt').to(device)

    # Read the log file using pandas into a dataframe
    df = pd.read_csv(path_train + '/log.csv')

    # Plot all the values with respect to the epochs
    df.plot(x='epoch', figsize=(15, 8))
    plt.savefig(path_train + '/metrics.jpg')

    # Read namefile of images test
    file_name = os.listdir(path_test + 'Images/')

    # Jaccard Index (IoU)
    jaccard = JaccardIndex(num_classes=8,  ignore_index=0, average='micro').to(device)

    # Listo of color for each class
    colors = [[0, 0, 0],  # Black
              [255, 0, 0],  # Red
              [0, 255, 0],  # Green
              [0, 0, 255],  # Blue
              [255, 255, 0],  # Yellow
              [0, 255, 255],  # Green
              [255, 0, 255],  # Blue
              [125, 125, 125],  # Yellow
              ]

    iou_mean = 0
    for img in file_name:
        image_path = path_test + 'Images/' + img
        target_path = path_test + 'Target/' + img.split('.')[0] + '.png'
        with open(image_path, "rb") as image_file, open(target_path, "rb") as target_file:
            image = Image.open(image_file)
            image = image.convert("RGB")
            target = Image.open(target_file)
            target = target.convert("L")

            transform_img = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
            transform_targ = transforms.Compose([transforms.Resize(512), transforms.PILToTensor()])
            image = transform_img(image).unsqueeze(0)
            target = transform_targ(target)
            # One hot encoding
            target_ohe = F.one_hot(target.to(torch.int64), num_classes=8).permute(0, 3, 1, 2).float().to(device)
            # target_ohe = target_ohe[:, 1:5, :, :]
        with torch.no_grad():
            img_inference = model(image.type(torch.cuda.FloatTensor))['out']

        iou_balloon = jaccard(img_inference, target_ohe.int()).item()
        iou_mean += iou_balloon
        print(str(img) + ' - IoU: '+ str(iou_balloon))

        if np.sum(img_inference.cpu().detach().numpy()) == 0:
            output_image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            output_image = np.zeros((512, 512, 3), dtype=np.uint8)
            for c in range(5):
                class_probs = img_inference.cpu().detach().numpy()[0][c] > 0.5
                class_color = np.repeat(np.expand_dims(class_probs, 2), 3, axis=2) * colors[c]
                output_image = np.maximum(output_image, class_color)

        # Convert target to numpy array and assign color to each class
        target_np = np.array(target)
        target_color = np.zeros((512, 512, 3), dtype=np.uint8)
        for c in range(1, 5):
            class_pixels = target_np == c
            class_color = np.repeat(np.expand_dims(class_pixels, 3), 3, axis=3) * colors[c]
            class_color = np.squeeze(class_color, 0)
            target_color = np.maximum(target_color, class_color)

        # Plot the input image and the predicted output
        fig = plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(image.squeeze(0).permute(1, 2, 0))
        plt.title('Image')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(output_image)
        plt.title('Segmentation Output')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(target_color)
        plt.title('Ground True')
        plt.axis('off')
        plt.savefig(path_train + '/' + img, bbox_inches='tight')
        plt.close(fig)

    iou_mean = iou_mean/(len(file_name))
    print('iou_mean: ', iou_mean)
    parameters = ['\n' + 'Inference Metrics.' + '\n',
                  'IoU Mean: ' + str(iou_mean) + '\n']
    with open(path_train + '/Parameters.txt', 'a') as f:
        f.writelines(parameters)
        f.close()


if __name__ == "__main__":
    main()