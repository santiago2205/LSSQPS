# Learning Semantic Segmentation with Query Points Supervision
This is research to see how weakly supervised semantic segmentation with superpixel performs.
To do it we selected a [balloon](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) 
dataset fully labeled to do the test.

## Prerequisites
We recommend the user run the code with the environment provided in this repository:

```
conda env create -f environment.yml
```

You need install *hers_superpixel* library. To do it, open a terminal in DAL-HERS folder and type the following:
```
make module
```
### DAL-HERS Model set-up
Download the pretrained Deep Affinity Learning (DAL) model from [here](https://drive.google.com/file/d/14-uaeMAihLdMepfZAth19T1pfZIoMcaE/view?usp=sharing) and put it under the ```DAL-HERS/pretrained``` folder

## Preprocessing
Run *pipeline.py* to generate the superpixel mask from the original image.
To run it type the following command in the main folder, where *nC* is the number of superpixels
and *nP* is the number of points used to select superpixels randomly.

```
python pipeline.py --nC 300 --nP 30 --edge True
```
In the following image, you can see in the flowchart how it creates the superpixel mask.

![](/flowchart.jpg)

## Train DeepLabV3

```
python main.py --data_directory Dataset --exp_directory train_output/000_Test --train_percent 1 --epochs 200 --batch_size 4
```
* --data_directory: Directory of the dataset
* --exp_directory: Directory of the DeepLabV3 output
* --train_percent: Percentage of image train used to train (0.1 - 1)
* --epoch: Number of epoch
* --batch_size: Batch size used to train

## Inference
```
python inference.py --data_directory Dataset --exp_directory train_output/000_Test
```
* --data_directory: Directory of the dataset
* --exp_directory: Directory of the DeepLabV3 output
