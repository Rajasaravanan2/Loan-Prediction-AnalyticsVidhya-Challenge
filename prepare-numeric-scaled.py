import numpy as np

from util import Dataset
from sklearn.preprocessing import scale

print("Loading data...")

train_num = Dataset.load_part('train', 'numeric_mean')
test_num = Dataset.load_part('test', 'numeric_mean')

print("Scaling...")

all_scaled = scale(np.vstack((train_num, test_num)))

print("Saving...")

Dataset.save_part_features('numeric_mean_scaled', Dataset.get_part_features('numeric'))
Dataset(numeric_mean_scaled=all_scaled[:train_num.shape[0]]).save('train')
Dataset(numeric_mean_scaled=all_scaled[train_num.shape[0]:]).save('test')

print("Done.")
