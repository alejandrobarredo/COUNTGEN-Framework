import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from tqdm import tqdm
import os


shoes_path = './datasets/shoesVsHandbags/Shoes/'
handbags_path = './datasets/shoesVsHandbags/Handbags/'
edges2shoes_path = './datasets/edges2shoes/val/'
edges2handbags_path = './datasets/edges2handbags/val/'

edges2shoes_files = os.listdir(edges2shoes_path)
edges2handbags_files = os.listdir(edges2handbags_path)

for file in tqdm(edges2shoes_files):
    img = image.imread(edges2shoes_path + file)
    shoe_img = img[:, 256:, :]
    image.imsave(shoes_path + file, shoe_img)

for file in tqdm(edges2handbags_files):
    img = image.imread(edges2handbags_path + file)
    handbag_img = img[:, 256:, :]
    image.imsave(handbags_path + file, shoe_img)