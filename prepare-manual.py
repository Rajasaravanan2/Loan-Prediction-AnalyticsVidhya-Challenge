import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import scale

from tqdm import tqdm
from util import Dataset

print("Loading data...")

idx = Dataset.load_part("train", 'id')

train_cat = pd.DataFrame(Dataset.load_part("train", 'categorical_mode'), columns=Dataset.get_part_features('categorical_mode'), index=idx)
train_num = pd.DataFrame(Dataset.load_part("train", 'numeric_mean'), columns=Dataset.get_part_features('numeric_mean'), index=idx)

train = pd.concat([train_cat,train_num],axis=1)

idx = Dataset.load_part("test", 'id')

test_cat = pd.DataFrame(Dataset.load_part("test", 'categorical_mode'), columns=Dataset.get_part_features('categorical_mode'), index=idx)
test_num = pd.DataFrame(Dataset.load_part("test", 'numeric_mean'), columns=Dataset.get_part_features('numeric_mean'), index=idx)


test = pd.concat([test_cat,test_num],axis=1)

all = train.append(test)
print(all.head())

custom = pd.DataFrame()

custom["inc/lAmount"] = all.ApplicantIncome / all.LoanAmount
custom["prop_area_LAmount"] = all.Property_Area * all.LoanAmount
custom["chist_loanamnt"] = all.Credit_History * all.LoanAmount
custom["chist_loanamnt_term"] = all.Credit_History * all.LoanAmount / all.Loan_Amount_Term
custom["allIncome"] = all.ApplicantIncome + all.CoapplicantIncome
custom["netIncome"] = custom["allIncome"] - (all.LoanAmount/all.Loan_Amount_Term)
custom["depend_chist"] = all.Dependents * all.Credit_History
custom["married_chist"] = all.Credit_History * all.Married
all_scaled = scale(custom)
#all_scaled = custom

train_custom = all_scaled[:train_cat.shape[0]]
test_custom = all_scaled[train_cat.shape[0]:]
print(train_cat.head())
print(test_custom.shape)
print(test_cat.shape)
print(list(custom.columns))
#
# train_cat_enc = sp.hstack(train_cat_enc, format='csr')
# test_cat_enc = sp.hstack(test_cat_enc, format='csr')

Dataset.save_part_features('custom', list(custom.columns))
Dataset(custom=train_custom).save('train')
Dataset(custom=test_custom).save('test')
#
# print("Done.")
