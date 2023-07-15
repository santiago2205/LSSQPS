import os
import torch
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from torchgeo.datasets import LandCoverAI, DeepGlobeLandCover
from PIL import Image, ImageDraw
import sys
import argparse
sys.path.insert(0, "/home/name/LSSQPS/src/Preprocess/DAL-HERS/pybuild")
sys.path.insert(0, "/home/name/LSSQPS/src/Preprocess/DAL-HERS/pybuild")
sys.path.append(os.path.abspath('/home/name/LSSQPS/src/Preprocess/DAL-HERS'))
from analysis_single_nC import *


# input arguments
parser = argparse.ArgumentParser(description='Create a mask from superpixel and points on a folder of images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrained', required=True, help='Path to the pretrained model, eg:/home/name/LSSQPS/src/Preprocess/DAL-HERS/pretrained/DAL_loss=bce-rgb_date=23Feb2021.tar')
parser.add_argument('--path_dataset', required=True, help='Path to dataset folder, eg: /home/name/LSSQPS/src/Preprocess/LandCoverAI/Dataset/')
parser.add_argument('--path_dataset_nn', required=True, help='Path to dataset NN folder, eg: /home/name/LSSQPS/src/NN/Dataset/')
parser.add_argument('--output_dir', default='DAL-HERS/output/', help='Path to output folder superpixel')
parser.add_argument('--output_suff', default='', help='suffix to the output file')
parser.add_argument('--edge', default=False, help='whether to incorporate edge information')
parser.add_argument('--nC', default=300, type=int, help='the number of desired superpixels')
parser.add_argument('--nP', default=30, type=int, help='the number of desired points')
parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='default device (CPU / GPU)')
args = parser.parse_args()


# Define a function to generate full masks from image masks
def draw_full_mask():
    # Set the output directory path for the full masks
    full_mask_path = args.path_dataset + 'Masks/'

    # Create the output directory if it doesn't already exist
    os.makedirs(full_mask_path, exist_ok=True)

    # Read the list of filenames from the directory where the image masks are stored
    file_name = os.listdir(args.path_dataset + 'raw_dataset/output/')

    # Loop through each image filename in the list
    for image in tqdm(file_name):
        # Check if the file has the ".jpg" extension
        if image.endswith(".jpg"):
            # Extract the base name of the file (without the extension)
            base_name = image.split('.')[0]

            # Read the corresponding image mask file with "_m.png" extension using the Pillow library's Image class
            img = Image.open(f'{args.path_dataset}raw_dataset/output/{base_name}_m.png').convert('L')

            # resized to 450 pixels
            w, h = img.size
            div = h / 450 if h >= w else w / 450
            newsize = (int(w / div), int(h / div))

            # Resize the mask image using the resize method of the Image class
            img = img.resize(newsize, resample=Image.Resampling.NEAREST)

            # Save the resized mask image
            img.save(f'{full_mask_path}{base_name}.png')


# Define a function to resize the original images to the same size as the superpixel images
def resize_origin2superpixel():
    # Set the input directory path for the original images
    origin_input_path = args.path_dataset + 'Images/'

    # Create the input directory if it doesn't already exist
    os.makedirs(origin_input_path, exist_ok=True)

    # Read the list of filenames from the directory where the image masks are stored
    file_name = os.listdir(args.path_dataset + 'raw_dataset/output/')

    # Loop through each image filename in the list
    for image in tqdm(file_name):
        # Check if the file has the ".jpg" extension
        if image.endswith('.jpg'):
            # Read the original image using the Pillow library's Image class
            img = Image.open(f'{args.path_dataset}raw_dataset/output/{image}').convert('RGB')

            # resized to 450 pixels
            w, h = img.size
            div = h / 450 if h >= w else w / 450
            newsize = (int(w / div), int(h / div))

            # Resize the original image using the resize method of the Image class
            img = img.resize(newsize)

            # Save the resized original image to the input directory
            img.save(f'{origin_input_path}{image}')


# Define a function to generate random points on the image labels
def points_image(number_points):
    # Create an empty dictionary to store the point data for each image
    img_point = {}

    # Read the list of filenames from the directory where the original images are stored
    file_name = os.listdir(args.path_dataset + 'Images/')

    # Loop through each image filename in the list
    for image in tqdm(file_name):
        # Read the image label for the current image using the Pillow library's Image class
        img_label = Image.open(args.path_dataset + 'Masks/' + image.split('.')[0] + '.png').convert('L')

        # Create nested dictionaries to store the point data for each label class (0-4)
        img_point[image] = {}
        img_point[image]['points_nothing'] = {}  # 0. BackGround
        img_point[image]['points_building'] = {}  # 1. Building
        img_point[image]['points_woodland'] = {}  # 2. Woodland
        img_point[image]['points_water'] = {}  # 3. Water
        img_point[image]['points_road'] = {}  # 4. Road

        # Generate random points on the image label for each label class (0-4) using the random_points function
        img_point[image]['points_nothing'], img_point[image]['points_building'], img_point[image]['points_woodland'], \
        img_point[image]['points_water'], img_point[image]['points_road'] = random_points(number_points, img_label)

    # Return the dictionary containing the point data for each image
    return img_point


def random_points(number, img_label):
    # Convert the input image label to a numpy array
    img_array = np.array(img_label)

    # Identify the coordinates of pixels belonging to each of the 5 classes in the image
    class1 = np.transpose(np.where(img_array == 1))
    class2 = np.transpose(np.where(img_array == 2))
    class3 = np.transpose(np.where(img_array == 3))
    class4 = np.transpose(np.where(img_array == 4))
    class5 = np.transpose(np.where(img_array == 5))

    # Determine the number of points to be randomly selected from each class
    unique_pixels, count = np.unique(img_array, return_counts=True)
    num_points = int(number/len(unique_pixels))

    # Create empty lists to store the randomly selected points for each class
    points_nothing = []
    points_building = []
    points_woodland = []
    points_water = []
    points_road = []

    # For each class, if the number of pixels in that class is greater than the number of points to be selected,
    # then randomly select the specified number of points from that class and store their coordinates as tuples in the appropriate list
    if len(class1) > num_points:
        points_nothing = [tuple(point[::-1]) for point in
                          class1[np.random.choice(class1.shape[0], num_points, replace=False), :]]
    if len(class2) > num_points:
        points_building = [tuple(point[::-1]) for point in
                           class2[np.random.choice(class2.shape[0], num_points, replace=False), :]]
    if len(class3) > num_points:
        points_woodland = [tuple(point[::-1]) for point in
                           class3[np.random.choice(class3.shape[0], num_points, replace=False), :]]
    if len(class4) > num_points:
        points_water = [tuple(point[::-1]) for point in
                        class4[np.random.choice(class4.shape[0], num_points, replace=False), :]]
    if len(class5) > num_points:
        points_road = [tuple(point[::-1]) for point in
                       class5[np.random.choice(class5.shape[0], num_points, replace=False), :]]

    # Return all the lists of randomly selected points
    return points_nothing, points_building, points_woodland, points_water, points_road


def draw_image_point(img_points):
    # Create a directory to store the images with points drawn on them, if it doesn't already exist
    image_point_path = args.path_dataset + 'Images_Points/'
    os.makedirs(image_point_path, exist_ok=True)

    # Get a list of all the image file names in the input directory
    file_name = os.listdir(args.path_dataset + 'Images/')

    # For each image, open it, create a new image with the same dimensions, and draw points of different colors
    # on it based on the points associated with each class of the image in img_points
    for image in tqdm(file_name):
        # Open the image and convert it to RGB mode
        img = Image.open(f'{args.path_dataset}Images/{image}').convert('RGB')

        # Create a new ImageDraw object to draw on the image
        draw = ImageDraw.Draw(img)

        # Draw the points for each class in a different color
        draw.point(img_points[image]['points_nothing'], fill='white')
        draw.point(img_points[image]['points_building'], fill='red')
        draw.point(img_points[image]['points_woodland'], fill='yellow')
        draw.point(img_points[image]['points_water'], fill='blue')
        draw.point(img_points[image]['points_road'], fill='purple')

        # Save the modified image to the directory created earlier, using the original image file name as a basis
        img.save(image_point_path + image.split('.')[0] + '.png')


def draw_mask_from_point(img_points):
    # Create directory to save CSV files
    save_csv_path = args.path_dataset + args.output_dir + str(args.nC) + '/Masks/csv/'
    os.makedirs(save_csv_path, exist_ok=True)

    # Create directory to save superpixel masks
    mask_superpixel_path = args.path_dataset + 'Masks_Superpixel/'
    os.makedirs(mask_superpixel_path, exist_ok=True)

    # Get list of image file names in the Images directory
    file_name = os.listdir(args.path_dataset + 'Images/')

    # Iterate over each image in the Images directory
    for image in tqdm(file_name):
        aux_class = {}

        # Open CSV file for the current image
        df_csv = pd.read_csv(
            args.path_dataset + args.output_dir + str(args.nC) + '/csv/' + image.split('.')[0] + '.csv', header=None)

        # Iterate over each point labeled as "nothing" in the current image
        for point in img_points[image]['points_nothing']:
            # Check if the label for the current point has already been encountered
            if str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1]) in aux_class.keys():
                # Increment the count of points labeled as "nothing" for the current label
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])]['points_nothing'] += 1
            else:
                # Add a new label to the dictionary and set the count of points labeled as "nothing" to 1
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])] = {'points_nothing': 1,
                                                                                       'points_building': 0,
                                                                                       'points_woodland': 0,
                                                                                       'points_water': 0,
                                                                                       'points_road': 0}

        # Iterate over each point labeled as "building" in the current image
        for point in img_points[image]['points_building']:
            # Check if the label for the current point has already been encountered
            if str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1]) in aux_class.keys():
                # Increment the count of points labeled as "building" for the current label
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])]['points_building'] += 1
            else:
                # Add a new label to the dictionary and set the count of points labeled as "building" to 1
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])] = {'points_nothing': 0,
                                                                                       'points_building': 1,
                                                                                       'points_woodland': 0,
                                                                                       'points_water': 0,
                                                                                       'points_road': 0}

        # Iterate over each point labeled as "building" in the current image
        for point in img_points[image]['points_building']:
            # Check if the label for the current point has already been encountered
            if str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1]) in aux_class.keys():
                # Increment the count of points labeled as "building" for the current label
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])]['points_building'] += 1
            else:
                # Add a new label to the dictionary and set the count of points labeled as "woodland" to 1
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])] = {'points_nothing': 0,
                                                                                       'points_building': 0,
                                                                                       'points_woodland': 1,
                                                                                       'points_water': 0,
                                                                                       'points_road': 0}

        # Iterate over each point labeled as "building" in the current image
        for point in img_points[image]['points_building']:
            # Check if the label for the current point has already been encountered
            if str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1]) in aux_class.keys():
                # Increment the count of points labeled as "building" for the current label
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])]['points_building'] += 1
            else:
                # Add a new label to the dictionary and set the count of points labeled as "water" to 1
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])] = {'points_nothing': 0,
                                                                                       'points_building': 0,
                                                                                       'points_woodland': 0,
                                                                                       'points_water': 1,
                                                                                       'points_road': 0}

        # Iterate over each point labeled as "building" in the current image
        for point in img_points[image]['points_building']:
            # Check if the label for the current point has already been encountered
            if str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1]) in aux_class.keys():
                # Increment the count of points labeled as "building" for the current label
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])]['points_building'] += 1
            else:
                # Add a new label to the dictionary and set the count of points labeled as "road" to 1
                aux_class[str(df_csv.values[int(point[1]) - 1, int(point[0]) - 1])] = {'points_nothing': 0,
                                                                                       'points_building': 0,
                                                                                       'points_woodland': 0,
                                                                                       'points_water': 0,
                                                                                       'points_road': 1}

        # iterate over each superpixel label in the input image
        for key in aux_class:
            # determine the maximum class label assigned to the current superpixel label
            max_class = max(aux_class[key], key=aux_class[key].get)

            # map the superpixel label to a corresponding class label
            if max_class == 'points_nothing':
                df_csv.replace(to_replace=int(key), value='a', inplace=True)
            elif max_class == 'points_building':
                df_csv.replace(to_replace=int(key), value='b', inplace=True)
            elif max_class == 'points_woodland':
                df_csv.replace(to_replace=int(key), value='c', inplace=True)
            elif max_class == 'points_water':
                df_csv.replace(to_replace=int(key), value='d', inplace=True)
            elif max_class == 'points_road':
                df_csv.replace(to_replace=int(key), value='e', inplace=True)

        # if there are more than two classes present in the image
        if len(aux_class) > 2:
            # replace the class labels with integer values
            list_supix = range(0, args.nC + 1)
            df_csv.replace(to_replace=list_supix, value=0, inplace=True)
            df_csv.replace(to_replace='a', value=1, inplace=True)
            df_csv.replace(to_replace='b', value=2, inplace=True)
            df_csv.replace(to_replace='c', value=3, inplace=True)
            df_csv.replace(to_replace='d', value=4, inplace=True)
            df_csv.replace(to_replace='e', value=5, inplace=True)

        # save the resulting image segmentation mask as a CSV file
        df_csv.to_csv(save_csv_path + image.split('.')[0] + '.csv', header=None, index=None)

        # load the CSV file as a numpy array
        my_data = np.genfromtxt(save_csv_path + image.split('.')[0] + '.csv', delimiter=',', dtype=np.uint8)

        # convert the numpy array to a PIL image and save it as a PNG file
        im_mask = Image.fromarray(my_data)
        im_mask.save(mask_superpixel_path + image.split('.')[0] + '.png')


def copy_split_file_in_neural_network():
    # Get a list of image files from the dataset directory
    image_list = np.array(os.listdir(args.path_dataset + 'Images/'))

    # Define paths for the directories to store training, validation and testing data for neural network
    deeplab_train_img_path = f'{args.path_dataset_nn}Train/Images/'
    deeplab_train_target_path = f'{args.path_dataset_nn}Train/Target/'
    deeplab_val_img_path = f'{args.path_dataset_nn}Val/Images/'
    deeplab_val_target_path = f'{args.path_dataset_nn}Val/Target/'
    deeplab_test_img_path = f'{args.path_dataset_nn}Test/Images/'
    deeplab_test_target_path = f'{args.path_dataset_nn}Test/Target/'

    # Create directories for training, validation and testing data if they don't exist
    os.makedirs(deeplab_train_img_path, exist_ok=True)
    os.makedirs(deeplab_train_target_path, exist_ok=True)
    os.makedirs(deeplab_val_img_path, exist_ok=True)
    os.makedirs(deeplab_val_target_path, exist_ok=True)
    os.makedirs(deeplab_test_img_path, exist_ok=True)
    os.makedirs(deeplab_test_target_path, exist_ok=True)

    # Initialize variables for processing images
    fraction = 0.4
    np.random.seed(100)
    indices = np.arange(len(image_list))
    np.random.shuffle(indices)
    image_list = image_list[indices]

    # Loop through the subsets of data (training, validation and testing)
    for subset in tqdm(['Train', 'Val', 'Test']):
        # Determine the images to be processed based on the subset
        if subset == 'Train':
            image_names = image_list[:int(np.ceil(len(image_list) * (1 - fraction)))]
        if subset == 'Val':
            image_names = image_list[int(np.ceil(len(image_list) * (1 - fraction))):int(
                np.ceil(len(image_list) * (1 - (fraction / 2))))]
        if subset == 'Test':
            image_names = image_list[int(np.ceil(len(image_list) * (1 - (fraction / 2)))):]

        # Copy image and mask files to the respective directories
        for file in image_names:
            # Copy image file to the respective directory
            shutil.copyfile(args.path_dataset + 'Images/' + file.split('.')[0] + '.jpg',
                            args.path_dataset_nn + subset + '/Images/' + file.split('.')[0] + '.jpg')

            # Copy mask file to the respective directory for testing subset
            if subset == 'Test':
                shutil.copyfile(args.path_dataset + 'Masks/' + file.split('.')[0] + '.png',
                                args.path_dataset_nn + subset + '/Target/' + file.split('.')[0] + '.png')
            # Copy mask file to the respective directory for training and validation subsets
            else:
                shutil.copyfile(args.path_dataset + 'Masks/' + file.split('.')[0] + '.png',
                                args.path_dataset_nn + subset + '/Target/' + file.split('.')[0] + '.png')

if __name__ == '__main__':
    print('Start')
    print('Download Dataset')
    dataset = LandCoverAI(root=args.path_dataset, download=True, checksum=True)
    print("Draw full mask")
    draw_full_mask()
    print("Resize image to superpixel size")
    resize_origin2superpixel()
    print("Superpixel")
    superpixel(args.nC, args.pretrained, args.path_dataset + 'Images/', args.path_dataset + args.output_dir,
               args.output_suff, args.edge, args.device)
    print("Generate random points")
    json_random_point = points_image(args.nP)
    print("Generate image with points")
    draw_image_point(json_random_point)
    print("Draw target from superpixel and points")
    draw_mask_from_point(json_random_point)
    print("Copy image and target in neural network")
    copy_split_file_in_neural_network()
    print('Finish')
