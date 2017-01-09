import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import LabelEncoder,scale

from tqdm import tqdm
from util import Dataset

print("Loading data...")

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

target = "Loan_Status"
id = "Loan_ID"
num_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
cat_columns = [c for c in train.columns if (c not in [*num_columns,target,id])]

n_train = train.shape[0]
all_data = train.append(test)

all_data.loc[((all_data.Gender == "Male") & (all_data.Married.isnull())),"Married"] = "Yes"
all_data.loc[((all_data.Gender == "Female") & (all_data.Married.isnull())),"Married"] = "No"

all_data.loc[((all_data.Married == "Yes") & (all_data.Gender.isnull())),"Gender"] = "Male"
all_data.loc[((all_data.Married == "No") & (all_data.Gender.isnull())),"Gender"] = "Female"

#all_data.loc[((all_data.Loan_Status == "N") & (all_data.Credit_History.isnull())),"Credit_History"] = 0.0
#all_data.loc[((all_data.Loan_Status == "Y") & (all_data.Credit_History.isnull())),"Credit_History"] = 1.0

all_data.Credit_History.fillna(1.0,inplace=True)
all_data["Dependents"] = all_data.groupby(["Married","Gender","Property_Area"]).Dependents.transform(lambda x: x.fillna(x.value_counts().idxmax()))

all_data["LoanAmount"] = all_data.groupby("Education").LoanAmount.transform(lambda x: x.fillna(x.mean()))
all_data["Loan_Amount_Term"] = all_data.groupby("Married").Loan_Amount_Term.transform(lambda x: x.fillna(x.mean()))

train = all_data.dropna(axis=0)
test = all_data[all_data.Self_Employed.isnull()]
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
model.fit(train[["LoanAmount","Loan_Amount_Term"]],train["Self_Employed"])
missing_se = model.predict(test[["LoanAmount","Loan_Amount_Term"]])

all_data.loc[all_data.Self_Employed.isnull(),"Self_Employed"] = missing_se

all_data["Loan_Amount_Term"] = all_data.groupby("Married").Loan_Amount_Term.transform(lambda x: x.fillna(x.mean()))

all_data[num_columns] = scale(all_data[num_columns])

train_cleaned = all_data[:n_train]
test_cleaned = all_data[n_train:]



train_cleaned.loc[:,cat_columns] = train_cleaned[cat_columns].apply(LabelEncoder().fit_transform)
test_cleaned.loc[:,cat_columns] = test_cleaned[cat_columns].apply(LabelEncoder().fit_transform)

Dataset(cat_manual=train_cleaned[cat_columns].values).save("train")
Dataset(num_manual=train_cleaned[num_columns].values.astype(np.float32)).save("train")
Dataset(id=train_cleaned[id]).save("train")

Dataset(cat_manual=test_cleaned[cat_columns].values).save("test")
Dataset(num_manual=test_cleaned[num_columns].values.astype(np.float32)).save("test")
Dataset(id=test_cleaned[id]).save("test")

Dataset.save_part_features('cat_manual', Dataset.get_part_features('categorical_mode'))
Dataset.save_part_features('num_manual', Dataset.get_part_features('numeric_mean'))


le = LabelEncoder()
le.fit(train_cleaned[target])
print(le.transform(train_cleaned[target]))
Dataset(target=le.transform(train_cleaned[target])).save("train")
Dataset(target_labels=le.classes_).save("train")




#
# train_cat_enc = sp.hstack(train_cat_enc, format='csr')
# test_cat_enc = sp.hstack(test_cat_enc, format='csr')

# Dataset.save_part_features('custom', list(custom.columns))
# Dataset(custom=train_custom).save('train')
# Dataset(custom=test_custom).save('test')
#
# print("Done.")
