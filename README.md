# lightweight_cnn_classifier

The python code lightweight_cnn_classifier is an image classifier which can be embedded/mobile friendly due to its small size.

==========================================================================
Lightweight CNN Classifier for Image Data
==========================================================================

lightweight_cnn_classifier v1.0

This python code is a foundation for a lightweight (e.g. embedded/mobile)  
CNN image classifier in PyTorch. It is a fully functioning code and can be
easiliy adapted for various classes (cars, flowers, etc.). The CNN can be
made more sophisticated by adding more convolutions, kernels, batchnorm, 
and dropout. By default it is set up for two classes and has performance of
up to 75 percent accuracy for validation data on the Kaggle flowers data set. 
This can be improved by cleaning and normalizing the dataset (which is 
needed). The code runs on CPU but plans to adapt to GPU exist. It is left 
for the user to download the dataset for experimentation. It can be 
found on Kaggle at:

https://www.kaggle.com/alxmamaev/flowers-recognition


IMPORTANT NOTE:
When organizing data in folders to be input to dataloader, 
it is important to keep in mind the following for correct loading:

(1) The train and val data were separated into their own folders by hand by 
class (rose and tulip) called 'flowers_datasets/training_sets' and 
'flowers_datasets/val_sets'. That means the sub-folder 'training_sets' 
contains two folders: rose and tulip. The same is true for validation data 
held in the folder 'flowers_datasets/val_sets'. So the organization looks like:

flowers_datasets > training_sets > rose, tulip
flowers_datasets > val_sets > rose, tulip

(2) The test data is organized differently since there are no labels 
for those images. Instead, the test data are held in the folder 
'flowers_datasets/test_sets' where the sub-folder here 'test_sets' 
just contains one folder called 'test'. This is instead of the rose and tulip 
folders. So the organization looks like:

flowers_datasets > test_sets > test

=============================================================================

