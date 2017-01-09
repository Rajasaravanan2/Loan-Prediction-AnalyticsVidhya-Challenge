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
    if name == 'train':
        num_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
        cat_columns = [c for c in data.columns if (c not in [*num_columns,target,id])]

        print(cat_columns)
        print(num_columns)
        Dataset.save_part_features('categorical_mode', cat_columns)
        Dataset.save_part_features('numeric_mean', num_columns)
        Dataset.save_part_features('numeric_median', num_columns)


    cData_mode = data[cat_columns].copy()
    cat_mode = cData_mode.mode().ix[0]
    cData_mode.fillna(cat_mode,inplace=True)

    cData_mode = cData_mode.apply(LabelEncoder().fit_transform)

    #print(cat_data.isnull().sum())
    numData_mean = data[num_columns].copy()
    num_mean = numData_mean.mean()
    numData_mean.fillna(num_mean,inplace=True)

    numData_median = data[num_columns].copy()
    num_meadian = numData_median.mean()
    numData_median.fillna(num_meadian,inplace=True)

    Dataset(categorical_mode=cData_mode.values).save(name)
    Dataset(numeric_mean=numData_mean.values.astype(np.float32)).save(name)
    Dataset(numeric_median=numData_median.values.astype(np.float32)).save(name)

    Dataset(id=data[id]).save(name)


    if target in data.columns:
        le = LabelEncoder()
        le.fit(data[target])
        print(le.transform(data[target]))
        Dataset(target=le.transform(data[target])).save(name)
        Dataset(target_labels=le.classes_).save(name)




print("Done.")
