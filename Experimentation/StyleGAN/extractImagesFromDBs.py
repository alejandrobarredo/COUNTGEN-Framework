from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy as np
import os
from os.path import exists, join
import matplotlib.pyplot as plt

db_path = './datasets/lmdb_dbs/'
datasets = os.listdir(db_path)

out_dir = './datasets/classifier_data/'

for dataset_path in datasets:
    class_name = dataset_path.split('_')[0]
    env = lmdb.open(db_path + dataset_path)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            img = cv2.imdecode(
                np.frombuffer(val, dtype=np.uint8), 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
            np.save(out_dir + class_name + '/' + class_name + '_' + str(
                count) + 'npy', img)
            count += 1
