# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:43:05 2021

@author: mbadal1996
"""
# ==========================================================================
# Lightweight CNN Classifier for Kaggle Flowers Data (or other image classes)
# ==========================================================================

# This python code is a foundation for a lightweight (e.g. embedded/mobile)  
# CNN image classifier in PyTorch. It is a fully functioning code and can be
# easiliy adapted for various classes (cars, flowers, etc.). The CNN can be
# made more sophisticated by adding more convolutions, kernels, batchnorm, 
# and dropout. By default it is set up for two classes and has performance of
# up to 75 percent accuracy for validation data on the Kaggle flowers data set. 
# This can be improved by cleaning and normalizing the dataset (which is 
# needed). The code runs on CPU but plans to adapt to GPU exist. It is left 
# for the user to download the dataset for experimentation. It can be 
# found on Kaggle at:

# https://www.kaggle.com/alxmamaev/flowers-recognition


# IMPORTANT NOTE:
# When organizing data in folders to be input to dataloader, 
# it is important to keep in mind the following for correct loading:

# (1) The train and val data were separated into their own folders by hand by 
# class (rose and tulip) called 'flowers_datasets/training_sets' and 
# 'flowers_datasets/val_sets'. That means the sub-folder 'training_sets' 
# contains two folders: rose and tulip. The same is true for validation data 
# held in the folder 'flowers_datasets/val_sets'. So the organization looks like:

# flowers_datasets > train_sets > rose, tulip
# flowers_datasets > val_sets > rose, tulip

# (2) The test data is organized differently since there are no labels 
# for those images. Instead, the test data are held in the folder 
# 'flowers_datasets/test_sets' where the sub-folder here 'test_sets' 
# just contains one folder called 'test'. This is instead of the rose and tulip 
# folders. So the organization looks like:

# flowers_datasets > test_sets > test

# =============================================================================

# Python
import numpy as np
import matplotlib.pyplot as plt

# Pytorch
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

# ===========================================================================
# Parameters

# Input number of classes (chosen by user)
num_class = 2

# Image Parameters
CH = 3  # number of channels
ratio = 1.5625  # choose width/height ratio for resizing input images
imagewidth = 157   # 157 for 100x100 # size of image dimension cropped to square 
imageheight = int(np.floor(imagewidth/ratio))
cropsize = imageheight  # square img is imageheight x imageheight

# Neural Net Parameters
learn_rate = 5e-4  # 1e-3 good
num_epochs = 5  # about 15-20 batches good; try 15 first
batch_size = 100  # batch sizes of 50 and 100 good; try 50 first

# Seed for reproduceable random numbers (eg weights and biases)
torch.manual_seed(1234)


# ======================================================================

# Create transforms for training data augmentation. In each epoch, random 
# transforms will be applied according to the Compose function. They are random 
# since we are explicitly choosing "Random" versions of the transforms. To "increase
# the dataset" one should run more epochs, since each epoch has new random data.
# NOTE: Augmentation should only be applied to Training data.
# NOTE: When using augmentation transforms, it is best to use larger batches

transform_train = transforms.Compose([
        transforms.Resize([imageheight, imagewidth]),
        transforms.CenterCrop(cropsize),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(degrees = (-20,20)), 
        #transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()])

transform_val = transforms.Compose([
        transforms.Resize([imageheight, imagewidth]),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor()])

transform_test = transforms.Compose([
        transforms.Resize([imageheight, imagewidth]),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor()])

# ----------------------------------------------------------
# Import train and validation data and set up data loader.
# Note that ImageFolder will organize data according to class labels of the 
# folders "roses, tulips, etc" as found in the train and val data folder.
# NOTE: When calling a specific image (such as 135) from train data, the first XXX
# images are class 0, then the next YYY are class 1, and etc.


# Training Data
images_train = datasets.ImageFolder('flowers_datasets/train_sets',transform=transform_train)
loader_train = torch.utils.data.DataLoader(images_train, shuffle=True,batch_size=batch_size)
# Validation Data
images_val = datasets.ImageFolder('flowers_datasets/val_sets',transform=transform_val)
loader_val = torch.utils.data.DataLoader(images_val, shuffle=True,batch_size=batch_size)
# Test Data
images_test = datasets.ImageFolder('flowers_datasets/test_sets',transform=transform_test)
loader_test = torch.utils.data.DataLoader(images_test, shuffle=False,batch_size=len(images_test))

# PLot sample image directly from data obtained by ImageFolder class
X_example,y_true_train = images_train[4]
plt.imshow(X_example.permute(1, 2, 0))  # Permute image dimensions to H x W x C (Channel) 
plt.title("Example of image with label "+str(y_true_train))
plt.show()

# print(images_train.classes)  # Output classes: 0, 1, 2, etc.

# ======================================================================

# Define CNN Model

# NOTE NOTE NOTE: This CNN takes input images of size 100x100 pixels

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pooling
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        # Convolution
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4)
        self.conv3 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4)
        self.conv4 = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4)
        # Fully Connected
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 100) # fully connected layer
        self.fc2 = torch.nn.Linear(100, 50)  # intermediate fully connected layer
        self.fc3 = torch.nn.Linear(50, 2)  # output fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 3 * 3)   
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)       
        return x


# ======================================================================


# Create instance of 'Net' class
model = Net()

# Choose the desired loss function.
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

# Choose the desired optimizer to train parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


# ======================================================================

# Initialize tensor to store loss values
result_vals = torch.zeros(num_epochs,4)
count_train = 0  # Initialize Counter
count_val = 0  # Initialize Counter
count_test = 0  # Initialize Counter

print(' ')
print('epoch     ave_loss_train         ave_loss_val')

for epoch in range(num_epochs):
    # New epoch begins
    
    # Initialize values
    running_loss_train = 0
    running_loss_val = 0
    num_batches_train = 0
    num_batches_val = 0
    count_train = 0
    count_val = 0
    
    model.train() # Set torch into train mode
    
    for X_train,y_true_train in loader_train:
        # (X,y) is a mini-batch:
        # (N: batch-size, 3: num channels, height x width)
        # X size N x 3 x cropsize x cropsize 
        # y size N
        
        # Reset gradients to zero for new batch
        optimizer.zero_grad()
        
        # Run model and compute loss
        N,C,nX,nY = X_train.size()  # Extract image/batch parameters
        y_pred_train = model(X_train.view(N,C,nX,nY))  # Evaluate model on batch
        loss_train = loss_fn(y_pred_train, y_true_train)  # Compute loss
        
        # Back propagation
        loss_train.backward()   
        
        # Update the NN parameters
        optimizer.step()
        
        # Update running loss after each batch
        running_loss_train += loss_train.detach().numpy()
        num_batches_train += 1 
        
        # Compute accuracy for train data per epoch
        for i in range(0,len(y_true_train)):
            # Compare y_pred_train to y_true_train
            if torch.argmax(y_pred_train[i,:]).item() == y_true_train[i].item():
                count_train = count_train + 1
    
    model.eval()  # Set torch into eval mode
    #with torch.no_grad():
    for X_val,y_true_val in loader_val:
        # (X,y) is a mini-batch:
        # (N: batch-size, 3: num channels, height x width)
        # X size N x 3 x cropsize x cropsize 
        # y size N
        
        # Run model and compute loss
        N,C,nX,nY = X_val.size()  # Extract image/batch parameters
        y_pred_val = model(X_val.view(N,C,nX,nY))  # Evaluate model on batch
        loss_val = loss_fn(y_pred_val, y_true_val)  # Compute loss
        
        # Update running loss after each batch
        running_loss_val += loss_val.detach().numpy()
        num_batches_val += 1 
        
        # Compute accuracy for train data per epoch
        for i in range(0,len(y_true_val)):
            # Compare y_pred_val to y_true_val
            if torch.argmax(y_pred_val[i,:]).item() == y_true_val[i].item():
                count_val = count_val + 1
    
    

    ave_loss_train = running_loss_train/num_batches_train
    ave_loss_val = running_loss_val/num_batches_val
    ave_accuracy_train = count_train/len(images_train)
    ave_accuracy_val = count_val/len(images_val)
    
    # Store loss values to tensor "loss_vals" for later plotting
    result_vals[epoch, 0] = ave_loss_train  # loss per epoch
    result_vals[epoch, 1] = ave_loss_val  # loss per epoch
    result_vals[epoch, 2] = ave_accuracy_train  # accuracy per epoch
    result_vals[epoch, 3] = ave_accuracy_val  # accuracy per epoch
    
    # Print loss every N epochs
    #if epoch % 5 == 4:
    print(epoch, '      ', ave_loss_train.item(), '  ', ave_loss_val.item())
    
    
# ==========================================================================
# Plot Loss and Accuracy for train and validation sets
# ==========================================================================

xvals = torch.linspace(0, num_epochs, num_epochs+1)
plt.plot(xvals[0:num_epochs].numpy(), result_vals[:,0].numpy())
plt.plot(xvals[0:num_epochs].numpy(), result_vals[:,1].numpy())
plt.legend(['loss_train', 'loss_val'], loc='upper right')
#plt.xticks(xvals[0:num_epochs])
plt.title('Loss (NN Classifier)')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.tick_params(right=True, labelright=True)
plt.show()


plt.plot(xvals[0:num_epochs].numpy(), result_vals[:,2].numpy())
plt.plot(xvals[0:num_epochs].numpy(), result_vals[:,3].numpy())
plt.legend(['accuracy_train', 'accuracy_val'], loc='lower right')
#plt.xticks(xvals[0:num_epochs])
plt.title('Accuracy (NN Classifier)')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.tick_params(right=True, labelright=True)
# plt.ylim(-0.15, 1.0)
plt.show()



# ===========================================================================


count_test = 0
test_predict = torch.zeros(len(images_test)).long()

model.eval()
for X_test,y_test in loader_test:
    y_pred_test = model(X_test.view(len(images_test),C,nX,nY))
    for i in range(0,len(images_test)):
        test_predict[i] = torch.argmax(y_pred_test[i,:]).item()
        # Compare y_pred_test to y_true_test
#        if torch.argmax(y_pred_test[i,:]).item() == y_true_test[i].item():
#            count_test = count_test + 1
        
print(' ')
print('Predicted classes from validation data (Rose = 0, Tulip = 1): ')
print(test_predict)
print(' ')
#print('Model accuracy in predicting test data:')
#print(count_test/len(y_true_test))
#print(' ')



# ==========================================================================

# Acquire Data from Batches for Confusion Matrices

count_train = 0
count_val = 0

predict_train = []  # Initialize
y_true_train_all = []  # Initialize
for X_train,y_true_train in loader_train:
    N,C,nX,nY = X_train.size()
    model.eval()
    y_pred_train = model(X_train.view(N,C,nX,nY))
    
    for i in range(0,len(y_true_train)):
        y_true_train_all.append(y_true_train[i])
        predict_train.append(torch.argmax(y_pred_train[i,:]).item())
        # Compare y_pred_train to y_true_train
        if torch.argmax(y_pred_train[i,:]).item() == y_true_train[i].item():
            count_train = count_train + 1


predict_val = []  # Initialize 
y_true_val_all = []  # Initialize
for X_val,y_true_val in loader_val:
    N,C,nX,nY = X_val.size()
    model.eval()
    y_pred_val = model(X_val.view(N,C,nX,nY))
    
    for i in range(0,len(y_true_val)):
        y_true_val_all.append(y_true_val[i])
        predict_val.append(torch.argmax(y_pred_val[i,:]).item())
        # Compare y_pred_val to y_true_val
        if torch.argmax(y_pred_val[i,:]).item() == y_true_val[i].item():
            count_val = count_val + 1

# Compute accuracy for train and val data using trained model
print(' ')
print('NN model accuracy in predicting training data:')
print(count_train/len(y_true_train_all))
print(' ')
print('NN model accuracy in predicting val data:')
print(count_val/len(y_true_val_all))


# ---------------------------------------------

# Initialize tensors to store confusion values
ZZ_train = torch.zeros(num_class,num_class).long()  
ZZ_val = torch.zeros(num_class,num_class).long()  

# Step through "True" and "Predicted" values and update confusion matrix
for j in range(0,len(y_true_train_all)):
    # For each "j" add +1 to ZZ at coordinate (predict_train[j],y_true_train[j])
    ZZ_train[predict_train[j],y_true_train_all[j]] += 1

for j in range(0,len(y_true_val_all)):
    # For each "j" add +1 to ZZ at coordinate (predict_val[j],y_true_val[j])
    ZZ_val[predict_val[j],y_true_val_all[j]] += 1 


# -------------------------------------------------------
# Create plotting function for confusion matrices

def confplot_func(input_data):

    # Create list of axis tick marks based on number of classes
    axes = []  # initialize list
    for i in range(0,num_class):
        axes.append(i)
        
    # Plot results of confusion matrix ZZ for Train and Validation data
    plt.imshow(input_data,extent=(-0.5,num_class-0.5,num_class-0.5,-0.5),
               interpolation='none',cmap='coolwarm')
    plt.xlabel('True Classes', fontsize = 12)
    plt.ylabel('Predicted Classes', fontsize = 12)
    plt.colorbar(fraction=0.145, pad=0.055, aspect=5.5)
    plt.xticks(axes, fontsize = 11)
    plt.yticks(axes, fontsize = 11)

    # Insert count text for each box in confusion matrix
    for i in range(0,num_class):
        for j in range(0,num_class):
            plt.text(j, i, format(input_data[i,j], 'd'), 
            horizontalalignment="center", verticalalignment="center", 
            color="white", fontsize = 16, fontweight = 'bold')

# -------------------------------------------------------

# Plot results of confusion matrix ZZ for Train and Validation data
plt.figure(figsize=(9,9))  # plot figures as subplots with given size

plt.subplot(1,2,1)
confplot_func(ZZ_train)  # pass train data to plotting function
plt.title('Confusion Matrix Train')

plt.subplot(1,2,2)
confplot_func(ZZ_val)  # pass validation data to plotting function
plt.title('Confusion Matrix Val')

plt.tight_layout(pad=1.5, w_pad=2.5, h_pad=1.0)

plt.show()

