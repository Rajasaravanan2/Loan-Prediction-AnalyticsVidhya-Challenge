import numpy as np
import scipy.sparse as sp
from scipy.stats import boxcox
import pandas as pd
from sklearn.preprocessing import scale

from tqdm import tqdm
from util import Dataset

print("Loading data...")

idx = Dataset.load_part("train", 'id')

train_num = pd.DataFrame(Dataset.load_part("train", 'numeric_mean'), columns=Dataset.get_part_features('numeric_mean'), index=idx)

idx = Dataset.load_part("test", 'id')

test_num = pd.DataFrame(Dataset.load_part("test", 'numeric_mean'), columns=Dataset.get_part_features('numeric_mean'), index=idx)


all_nData = train_num.append(test_num)
print(all_nData.head())

all_num_norm = pd.DataFrame()
all_num_norm["ApplicantIncome"] = np.log1p(all_nData.ApplicantIncome)
all_num_norm["CoapplicantIncome"] = np.log1p(all_nData.CoapplicantIncome)
all_num_norm["LoanAmount"] = (np.log1p(all_nData.LoanAmount))
all_num_norm["Loan_Amount_Term"] = np.log1p(all_nData.Loan_Amount_Term)

train_custom = all_num_norm[:train_num.shape[0]]
test_custom = all_num_norm[train_num.shape[0]:]
print(train_num.head())
print(test_custom.shape)
print(test_num.shape)
print(list(all_num_norm.columns))
#
# train_cat_enc = sp.hstack(train_cat_enc, format='csr')
# test_cat_enc = sp.hstack(test_cat_enc, format='csr')

Dataset.save_part_features('num_log1', list(all_num_norm.columns))
Dataset(num_log1=train_custom).save('train')
Dataset(num_log1=test_custom).save('test')
#
# print("Done.")
