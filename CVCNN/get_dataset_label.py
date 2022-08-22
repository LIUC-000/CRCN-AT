import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset(num):
    x = np.load(f"/data1/liuc/PycharmProjects/SEI/Dataset/X_train_{num}Class.npy")
    y = np.load(f"/data1/liuc/PycharmProjects/SEI/Dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.1, random_state=30)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.5, random_state=30)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def TrainDatasetKShotRound(num, k, RANDOM_SEED):
    x, X_val, X_test, y, Y_val, Y_test = TrainDataset(num)
    random_index_shot = []
    for i in range(num):
        index_shot = [index for index, value in enumerate(y) if value == i]
        random.seed(RANDOM_SEED)
        random_index_shot += random.sample(index_shot, k)
    random.shuffle(random_index_shot)
    X_train_K_Shot = x[random_index_shot, :, :]
    Y_train_K_Shot = y[random_index_shot]
    return X_train_K_Shot, X_val, X_test, Y_train_K_Shot, Y_val, Y_test

def TestDataset(num):
    x = np.load(f"/data1/liuc/PycharmProjects/SEI/Dataset/X_test_{num}Class.npy")
    y = np.load(f"/data1/liuc/PycharmProjects/SEI/Dataset/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    return x, y
