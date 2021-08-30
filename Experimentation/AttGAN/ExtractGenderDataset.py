import os
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import pandas as pd
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""

dataset_path = '/home/xai/Descargas/img_align_celeba_png/'
attributes_path = './datasets/list_attr_celeba.csv'

atts = pd.read_csv(attributes_path, sep=' ')
file_numbers = list(range(atts.shape[0]))
random.shuffle(file_numbers)
f_count = 0
m_count = 0

for file_n in tqdm(file_numbers):
    if f_count == 500 and m_count == 500:
        quit()
    else:
        if atts.iloc[file_n, 21] == 1 and m_count < 500:
            img = Image.open(dataset_path + atts.iloc[file_n, 0][:-4] + '.png')
            img = img.resize((128, 128))
            img = np.array(img) / 255.0
            name = atts.iloc[file_n, 0][:-4]
            np.save('./datasets/classifier_data/Male/' + name + '.npy', img)
            m_count += 1
        elif atts.iloc[file_n, 21] == -1 and f_count < 500:
            img = Image.open(dataset_path + atts.iloc[file_n, 0][:-4] + '.png')
            img = img.resize((128, 128))
            img = np.array(img) / 255.0
            name = atts.iloc[file_n, 0][:-4]
            np.save('./datasets/classifier_data/Female/' + name + '.npy', img)
            f_count += 1

