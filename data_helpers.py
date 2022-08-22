import os
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython.core.pylabtools import figsize
import torch
import torchvision
from torchvision import datasets, transforms

SOURCE_DATA_DIR = "data"
DEST_DATA_DIR = "data/splitted"
TRAINING_DIR = os.path.join(DEST_DATA_DIR,"training")
VALIDATION_DIR = os.path.join(DEST_DATA_DIR,"validation")

def make_data_split(split_size = 0.8):

    #create training dir
    if not os.path.isdir(TRAINING_DIR):
        os.mkdir(TRAINING_DIR)

    #create 1 in training
    c1_training_dir = os.path.join(TRAINING_DIR,"1")
    if not os.path.isdir(c1_training_dir):
        os.mkdir(c1_training_dir)

    #create 0 in training
    c0_training_dir = os.path.join(TRAINING_DIR,"0")
    if not os.path.isdir(c0_training_dir):
        os.mkdir(c0_training_dir)

    #create validation dir
    if not os.path.isdir(VALIDATION_DIR):
        os.mkdir(VALIDATION_DIR)

    #create 1 in validation
    c1_validation_dir = os.path.join(VALIDATION_DIR,"1")
    if not os.path.isdir(c1_validation_dir):
        os.mkdir(c1_validation_dir)

    #create 0 in validation
    c0_validation_dir = os.path.join(VALIDATION_DIR,"0")
    if not os.path.isdir(c0_validation_dir):
        os.mkdir(c0_validation_dir)


    c1_fp = os.path.join(SOURCE_DATA_DIR,"1","*.jpg")
    c0_fp = os.path.join(SOURCE_DATA_DIR,"0","*.jpg")
    c1_imgs_size = len(glob.glob(c1_fp))
    c0_imgs_size = len(glob.glob(c0_fp))

    print(c1_fp, c0_fp)
    print("copied c1: {}, c0: {}".format(c1_imgs_size, c0_imgs_size))

    for i,img in enumerate(glob.glob(c1_fp)):
        if i < (c1_imgs_size * split_size):
            shutil.copy(img,c1_training_dir)
        else:
            shutil.copy(img,c1_validation_dir)

    for i,img in enumerate(glob.glob(c0_fp)):
        if i < (c0_imgs_size * split_size):
            shutil.copy(img,c0_training_dir)
        else:
            shutil.copy(img,c0_validation_dir)

def sample_raw_images():

    c1_training_dir = os.path.join(TRAINING_DIR,"1")
    c0_training_dir = os.path.join(TRAINING_DIR,"0")

    samples_c1 = [os.path.join(c1_training_dir,np.random.choice(os.listdir(c1_training_dir),1)[0]) for _ in range(2)]
    samples_c0 = [os.path.join(c0_training_dir,np.random.choice(os.listdir(c0_training_dir),1)[0]) for _ in range(2)]

    nrows = 2
    ncols = 2

    fig, ax = plt.subplots(nrows,ncols,figsize = (10,10))
    ax = ax.flatten()

    for i in range(nrows*ncols):
        if i < 2:
            pic = plt.imread(samples_c1[i%2])
            ax[i].imshow(pic)
            ax[i].set_axis_off()
        else:
            pic = plt.imread(samples_c0[i%2])
            ax[i].imshow(pic)
            ax[i].set_axis_off()
    plt.show()

def vis_loader(dataloader):

    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(nrows,ncols,figsize = (10,10))
    ax = ax.flatten()

    for i in range(4):

        images = next(iter(dataloader))
        pic = np.transpose(images[0][0].numpy(), (1, 2, 0))
        classind = images[1][0].tolist()

        print(classind)
        ax[i].imshow(pic)
        ax[i].set_axis_off()
    plt.show()


def dataloaders():

    #transformations
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),                                
                                        torchvision.transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])

    #datasets
    train_data = datasets.ImageFolder(TRAINING_DIR,transform=train_transforms)
    test_data = datasets.ImageFolder(VALIDATION_DIR,transform=test_transforms)

    #dataloader
    trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=4)
    testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=4)

    return trainloader, testloader