import torch
import torch.optim as optim
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import argparse
import json
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import os
from datetime import datetime
from shutil import copyfile
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace
import random
import cv2


def load_brats(dataset_folder="./brats2021/brats_2021_2D", modality='t1'):

    cancerous_img = []
    for image in os.listdir(dataset_folder + '/cancerous'):
        if image[:5] == "BraTS":
            img_name = dataset_folder + '/cancerous/' + image + "/" + image + "_" + modality+ ".npy"
            cancerous_img.append(cv2.resize(np.load(img_name)[22:-22, 24:-20], dsize=(128, 128), interpolation=cv2.INTER_CUBIC))



    healthy_img = []
    for image in os.listdir(dataset_folder + '/healthy'):
        if image[:5] == "BraTS":
            img_name = dataset_folder + '/healthy/' + image + "/" + image + "_" + modality+ ".npy"
            healthy_img.append(cv2.resize(np.load(img_name)[22:-22, 24:-20], dsize=(128, 128), interpolation=cv2.INTER_CUBIC))


    print('cancerous_img : ', len(cancerous_img))


    random.shuffle(healthy_img)
    random.shuffle(cancerous_img)

    healthy_eval = healthy_img[:64]
    healthy_train = healthy_img[64:]

    cancerous_eval = cancerous_img[:64]
    cancerous_train = cancerous_img[64:]

    healthy_train = np.concatenate([healthy_train, healthy_train, healthy_train, healthy_train])


    cancerous_train = np.array(cancerous_train)

    cancerous_eval = np.array(cancerous_eval)
    healthy_eval = np.array(healthy_eval)

    img_train = np.concatenate([healthy_train, cancerous_train])

    img_eval = np.concatenate([healthy_eval, cancerous_eval])

    total_labels_train = np.concatenate([
    np.zeros(healthy_train.shape[0]),
    np.ones(cancerous_train.shape[0])])

    total_labels_eval = np.concatenate([
    np.zeros(healthy_eval.shape[0]),
    np.ones(cancerous_eval.shape[0])])

    print(img_train.shape)

    print(total_labels_eval.shape)

    print(total_labels_train.shape)

    np.save("./brats2021/brats_" + modality + "_img_train.npy", img_train)
    np.save("./brats2021/brats_" + modality + "_labels_train.npy", total_labels_train)

    np.save("./brats2021/brats_" + modality + "_img_eval.npy", img_eval)
    np.save("./brats2021/brats_" + modality + "_labels_eval.npy", total_labels_eval)

if __name__ == "__main__":


    load_brats()
