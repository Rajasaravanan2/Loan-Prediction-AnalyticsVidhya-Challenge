import numpy as np
import pandas as pd
from scipy.stats import skew, boxcox
from sklearn.preprocessing import scale

from tqdm import tqdm
from util import Dataset

import itertools

print("Loading data...")

train_num = Dataset.load_part('train', 'numeric_mean')
test_num = Dataset.load_part('test', 'numeric_mean')

train_cat = Dataset.load_part('train', 'categorical_dummy')
test_cat = Dataset.load_part('test', 'categorical_dummy')
ntrain = train_num.shape[0]

train = np.hstack([train_num,train_cat.toarray()])
test = np.hstack([test_num,test_cat.toarray()])


train_test = np.vstack([train,test])
print(train_test[0])
num_features = Dataset.get_part_features('numeric_mean')
cat_features = Dataset.get_part_features('categorical_dummy')

num_features.extend(cat_features)
print(num_features)
num_comb_df = pd.DataFrame()

with tqdm(total=train_num.shape[1], desc='  Transforming', unit='cols') as pbar:
    for comb in itertools.combinations(num_features, 2):
        feat = comb[0] + "_" + comb[1]
        #print(feat)
        num_comb_df[feat] = train_test[:,num_features.index(comb[0])-1] * train_test[:,num_features.index(comb[1])-1]
        print('Combining Columns:', feat)


print("Saving...")
print(num_comb_df.shape)
Dataset.save_part_features('all_comb', list(num_comb_df.columns))

train_num_comb = num_comb_df[:ntrain]
test_num_comb = num_comb_df[ntrain:]
num_comb_df = scale(num_comb_df)


Dataset(all_comb=train_num_comb.values).save('train')
Dataset(all_comb=test_num_comb.values).save('test')

print("Done.")


