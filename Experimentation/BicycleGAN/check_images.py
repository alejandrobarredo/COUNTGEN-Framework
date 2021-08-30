import pickle as pkl
import matplotlib.pyplot as plt

with open('./imgs/edges2shoes/test_images-0.pkl', 'rb') as f:
    real_A, real_B, fake_B = pkl.load(f)


print()