import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import scale,LabelEncoder

from tqdm import tqdm
from util import Dataset

print("Loading data...")
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
ntrain = train.shape[0]

train = train.drop(["Loan_ID","Loan_Status"],axis=1)
test = test.drop(["Loan_ID"],axis=1)

all_s = train.append(test)
all_s.Credit_History.fillna(2.0,inplace=True)
all_s["LoanAmount"] = all_s.groupby("Education").LoanAmount.transform(lambda x: x.fillna(x.mean()))
all_s["Loan_Amount_Term"] = all_s.groupby("Married").Loan_Amount_Term.transform(lambda x: x.fillna(x.mean()))


all_s.Self_Employed.fillna("X",inplace=True)
all_s.Dependents.fillna("X",inplace=True)

all_s.loc[((all_s.Gender == "Male") & (all_s.Married.isnull())),"Married"] = "No"
all_s.loc[((all_s.Gender == "Female") & (all_s.Married.isnull())),"Married"] = "Yes"

all_s["Property_Area"] = LabelEncoder().fit_transform(all_s["Property_Area"])
all_s["Married"] = LabelEncoder().fit_transform(all_s["Married"])
all_s["Self_Employed"] = LabelEncoder().fit_transform(all_s["Self_Employed"])
all_s["Dependents"] = LabelEncoder().fit_transform(all_s["Dependents"])


all_s["app_ch"] = all_s["ApplicantIncome"] * all_s["Credit_History"]
all_s["ci_ch"] = all_s["CoapplicantIncome"] * all_s["Credit_History"]
all_s["la_ch"] = all_s["LoanAmount"] * all_s["Credit_History"]
all_s["lat_ch"] = all_s["Loan_Amount_Term"] * all_s["Credit_History"]

all_s["cr_prop"] = all_s["Credit_History"] * all_s["Property_Area"]

# all_s["app_ch_p"] = all_s["ApplicantIncome"] * all_s["Credit_History"] * all_s["Property_Area"]
# all_s["ci_ch_p"] = all_s["CoapplicantIncome"] * all_s["Credit_History"] * all_s["Property_Area"]
# all_s["la_ch_p"] = all_s["LoanAmount"] * all_s["Credit_History"] * all_s["Property_Area"]
# all_s["lat_ch_p"] = all_s["Loan_Amount_Term"] * all_s["Credit_History"] * all_s["Property_Area"]
#


# all_s["app_s"] = all_s["ApplicantIncome"] * all_s["Self_Employed"]
# all_s["ci_s"] = all_s["CoapplicantIncome"] * all_s["Self_Employed"]
# all_s["la_s"] = all_s["LoanAmount"] * all_s["Self_Employed"]
# all_s["lat_s"] = all_s["Loan_Amount_Term"] * all_s["Self_Employed"]

features_to_drop = ['Gender', 'Married', 'Dependents','Education', 'Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']

all_filtered = all_s.drop(features_to_drop,axis=1)

print(train.columns)
# ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

train_cust = all_filtered[:ntrain]
test_cust = all_filtered[ntrain:]

print(all_s.columns)
#
# train_cat_enc = sp.hstack(train_cat_enc, format='csr')
# test_cat_enc = sp.hstack(test_cat_enc, format='csr')

Dataset.save_part_features('fSelect', list(all_filtered.columns))
Dataset(fSelect=train_cust.values).save('train')
Dataset(fSelect=test_cust.values).save('test')
#
# print("Done.")
