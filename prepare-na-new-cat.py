import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from util import Dataset

for name in ['train', 'test']:
    print("Processing %s..." % name)
    data = pd.read_csv('input/%s.csv' % name)
    target = "Loan_Status"
    id = "Loan_ID"
    # Save column names

    num_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
    cat_columns = [c for c in data.columns if (c not in [*num_columns,target,id])]



    cData_mode = data[cat_columns].copy()
    cData_mode.Credit_History.fillna(1.0,inplace=True)
    cData_mode.fillna("X",inplace=True)

    cData_mode = cData_mode.apply(LabelEncoder().fit_transform)


    Dataset.save_part_features('categorical_na_new', Dataset.get_part_features('categorical_mode'))
    Dataset(categorical_na_new=cData_mode.values).save(name)

    Dataset(id=data[id]).save(name)


    if target in data.columns:
        le = LabelEncoder()
        le.fit(data[target])
        print(le.transform(data[target]))
        Dataset(target=le.transform(data[target])).save(name)
        Dataset(target_labels=le.classes_).save(name)




print("Done.")
