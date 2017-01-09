import numpy as np
import pandas as pd
import xgboost as xgb

import argparse
import os
import datetime
import itertools

from shutil import copy2
from scipy import stats
np.random.seed(1337)

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import dump_svmlight_file
from sklearn.utils import shuffle, resample
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint
from keras import regularizers
#from keras_util import ExponentialMovingAverage, batch_generator

from statsmodels.regression.quantile_regression import QuantReg

from pylightgbm.models import GBMRegressor

from scipy.stats import boxcox

from bayes_opt import BayesianOptimization

from util import Dataset, load_prediction, hstack


categoricals = Dataset.get_part_features('categorical')


class Sklearn(BaseAlgo):

    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        self.model.fit(X_train, y_train)

        # if X_eval is not None and hasattr(self.model, 'staged_predict'):
        #     for i, p_eval in enumerate(self.model.staged_predict(X_eval)):
        #         print("Iter %d score: %.5f" % (i, eval_func(y_eval, p_eval)))

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func, seed=42):
        def fun(**params):
            for k in params:
                if type(param_grid[k][0]) is int:
                    params[k] = int(params[k])

            print("Trying %s..." % str(params))

            self.model.set_params(**params)
            self.fit(X_train, y_train)

            if hasattr(self.model, 'staged_predict'):
                best_score = 0
                best_i = -1
                for i, p_eval in enumerate(self.model.staged_predict(X_eval)):
                    mae = eval_func(y_eval, p_eval)

                    if mae < best_score:
                        best_score = mae
                        best_i = i

                print("Best score after %d iters: %.5f" % (best_i, best_score))
            else:
                p_eval = self.predict(X_eval)
                best_score = eval_func(y_eval, p_eval)

                print("Score: %.5f" % best_score)

            return best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params']))

    def grid_search(self, X_train, y_train, param_grid, eval_func, seed=42):

        gsearch = GridSearchCV(self.model, param_grid,verbose=10,cv=10)
        gsearch.fit(X_train,y_train)

        print(gsearch.best_params_)
        #print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params']))




def load_x(ds, preset):
    feature_parts = [Dataset.load_part(ds, part) for part in preset.get('features', [])]
    prediction_parts = [load_prediction(ds, p, mode=preset.get('predictions_mode', 'fulltrain')) for p in preset.get('predictions', [])]
    prediction_parts = [p.clip(lower=0.1).values.reshape((p.shape[0], 1)) for p in prediction_parts]

    if 'prediction_transform' in preset:
        prediction_parts = map(preset['prediction_transform'], prediction_parts)

    return hstack(feature_parts + prediction_parts)


def extract_feature_names(preset):
    x = []

    for part in preset.get('features', []):
        x += Dataset.get_part_features(part)

    lp = 1
    for pred in preset.get('predictions', []):
        if type(pred) is list:
            x.append('pred_%d' % lp)
            lp += 1
        else:
            x.append(pred)

    return x



norm_y_lambda = 0.7


def norm_y(y):
    return boxcox(np.log1p(y), lmbda=norm_y_lambda)


def norm_y_inv(y_bc):
    return np.expm1((y_bc * norm_y_lambda + 1)**(1/norm_y_lambda))


y_norm = (norm_y, norm_y_inv)
y_log = (np.log, np.exp)

le = LabelEncoder()

def en_y(y):
    le.fit(y)
    return le.transform(y)

def dc_y(y):
    return le.inverse_transform(y)

y_en_dc = (en_y,dc_y)

###  Target decoder
def y_decode(y):
    og = Dataset.load_part("train","target_labels")
    le = LabelEncoder()
    le.classes_ = og
    z = [int(i) for i in y]
    return le.inverse_transform(z)


from itertools import chain

def mode_agg(y,axis):

    modes =  stats.mode(y,axis=axis)[0]
    return list(chain.from_iterable(modes))



## Main part


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')
parser.add_argument('--optimize', action='store_true', help='optimize model params')
parser.add_argument('--fold', type=int, help='specify fold')
parser.add_argument('--threads', type=int, default=4, help='specify thread count')


args = parser.parse_args()

Xgb.default_params['nthread'] = args.threads
LightGBM.default_params['num_threads'] = args.threads

n_folds = 10


l1_predictions = [
'20170102-1903-et-0.80945',
'20170102-1905-rf-0.81107',
'20170102-1918-svc-0.68730',
'20170102-1921-lr-0.81107',
'20170102-1936-nb-0.79967',
'20170102-1938-nb1-0.75570',
'20170102-1944-knn-0.80619',
'20170102-2109-gbm-0.80456',
'20170102-2134-gbm1-0.80293'

]

l2_predictions = [

]

presets = {
    'et':{
        'features': ['numeric_mean', 'categorical_dummy'],
        'model': Sklearn(ExtraTreesClassifier(n_estimators=300,max_depth=3)),
        'n_bags': 10,
        'param_grid' : {'n_estimators' : [10,20,40,80,150,300,500],'max_depth' : [1,2,3,4,5,6,7,8,9,10]},
        'opt_method' : "grid_search"
    },

    'rf':{
        'features': ['numeric_mean', 'categorical_mode'],
        'model': Sklearn(RandomForestClassifier(n_estimators=150,max_depth=4)),
        'n_bags': 10,
        'param_grid': {'n_estimators' : [10,20,40,80,150,300,500],'max_depth' : [1,2,3,4,5,6,7,8,9,10]},
        'opt_method': "grid_search"
    },

    'svc':{
        'features': ['categorical_dummy'],
        'model': Sklearn(SVC()),
        'n_bags': 10,
        'param_grid': {'kernel': ['linear', 'rbf'], 'C' : [1, 10]},
        'opt_method': "grid_search"
    },

    'lr':{
        'features': ['numeric_mean','categorical_dummy'],
        'model': Sklearn(LogisticRegression(C=0.7)),
        'n_bags': 10,
        'param_grid': {'C' : [0.65,0.7,0.75]},
        'opt_method': "grid_search"
    },

    'nb':{
        'features': ['numeric_mean_scaled','categorical_dummy'],
        'model': Sklearn(BernoulliNB(class_prior=[192/(422+192),422/(422+192)])),
        'n_bags': 10,
        'param_grid': {'alpha' : np.arange(0.001,0.01,0.001)},
        'opt_method': "grid_search"

    },

    'nb1':{
        'features': ['numeric_mean_rank_norm','categorical_dummy'],
        'model': Sklearn(Pipeline([('to_dense', DenseTransformer()), ('nb', GaussianNB([192/(422+192),422/(422+192)]))]))
    },

    'knn':{
        'features': ['numeric_mean_rank_norm','categorical_dummy'],
        'model': Sklearn(KNeighborsClassifier(n_neighbors=21)),
        'n_bags': 10,
        'param_grid': {'n_neighbors' : np.arange(16,48,1)},
        'opt_method': "grid_search"

    },

    'gbm':{
        'features': ['numeric_mean','categorical_dummy'],
        'model': Sklearn(Pipeline([('to_dense', DenseTransformer()),('gbm', GradientBoostingClassifier(learning_rate=0.001,n_estimators=416))])),
        'n_bags': 10,
        'param_grid': {'gbm__learning_rate' : [0.001],'gbm__n_estimators' : np.arange(400,451,1)},
        'opt_method': "grid_search"
    },

    'gbm1':{
        'features': ['numeric_mean_boxcox','categorical_mode'],
        'model': Sklearn(GradientBoostingClassifier(learning_rate=0.001,n_estimators=420)),
        'n_bags': 10,
        'param_grid': {'learning_rate' : [0.001,0.007,0.001],'n_estimators' : np.arange(400,460,5)},
        'opt_method': "grid_search"
    },

    'lr-cust':{
        'features': ['numeric_mean','categorical_dummy','custom'],
        'model': Sklearn(LogisticRegression(C=0.7)),
        'n_bags': 10,
        'param_grid': {'C' : [0.1,1,0.1]},
        'opt_method': "grid_search"
    },

    'rf-l2':{
        'features': ['numeric_mean_scaled', 'categorical_dummy'],
        'predictions':l1_predictions,
        'model': Sklearn(Pipeline([('et_l2', ExtraTreesClassifier(n_estimators=100,max_depth=4))])),
        'n_bags': 10,
        'param_grid': {'et_l2__n_estimators' : np.arange(10,100,10),'et_l2__max_depth' : np.arange(1,10,1)},
        'opt_method': "grid_search"
    },

    'lr-l2':{
        'predictions':l1_predictions,
        'model': Sklearn(LogisticRegression(C=0.03)),
        'n_bags': 10,
        'param_grid': {'C' : np.arange(0.01,0.1,0.01)},
        'opt_method': "grid_search"
    },

    'nb-l2':{
        'features': ['numeric_mean_scaled', 'categorical_dummy'],
        'predictions':l1_predictions,
        'model': Sklearn(BernoulliNB()),
        'n_bags': 5,
        'param_grid': {'alpha' : np.arange(0.001,0.01,0.001)},
        'opt_method': "grid_search"
    },


}

print("Preset: %s" % args.preset)

preset = presets[args.preset]

feature_builders = preset.get('feature_builders', [])

n_bags = preset.get('n_bags', 1)
n_splits = preset.get('n_splits', 1)

y_aggregator = preset.get('agg', mode_agg)
y_transform, y_inv_transform = preset.get('y_transform', (lambda y: y, lambda y: y))

print("Loading train data...")
train_x = load_x('train', preset)
train_y = Dataset.load_part('train', 'target')
train_p = np.zeros((train_x.shape[0], n_splits * n_bags))
train_r = Dataset.load('train', parts=np.unique(sum([b.requirements for b in feature_builders], ['target'])))

feature_names = extract_feature_names(preset)
print(args.optimize)

if args.optimize:
    opt_train_idx, opt_eval_idx = train_test_split(range(len(train_y)), test_size=0.2)

    opt_train_x = train_x[opt_train_idx]
    opt_train_y = train_y[opt_train_idx]
    opt_train_r = train_r.slice(opt_train_idx)

    opt_eval_x = train_x[opt_eval_idx]
    opt_eval_y = train_y[opt_eval_idx]
    opt_eval_r = train_r.slice(opt_eval_idx)

    if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
        print("    Building per-fold features...")

        opt_train_x = [opt_train_x]
        opt_eval_x = [opt_eval_x]

        for fb in feature_builders:
            opt_train_x.append(fb.fit_transform(opt_train_r))
            opt_eval_x.append(fb.transform(opt_eval_r))

        opt_train_x = hstack(opt_train_x)
        opt_eval_x = hstack(opt_eval_x)

    if preset['opt_method'] != "grid_search":
        preset['model'].optimize(opt_train_x, y_transform(opt_train_y), opt_eval_x, y_transform(opt_eval_y), preset['param_grid'], eval_func=lambda yt, yp: accuracy_score(y_inv_transform(yt), y_inv_transform(yp)))
    else:
        preset['model'].grid_search(train_x, y_transform(train_y), preset['param_grid'], eval_func=lambda yt, yp: accuracy_score(y_inv_transform(yt), y_inv_transform(yp)))


print("Loading test data...")
test_x = load_x('test', preset)
test_r = Dataset.load('test', parts=np.unique([b.requirements for b in feature_builders]))
test_foldavg_p = np.zeros((test_x.shape[0], n_splits * n_bags * n_folds))
test_fulltrain_p = np.zeros((test_x.shape[0], n_bags))

if 'powers' in preset:
    print("Adding power features...")

    train_x, feature_names = add_powers(train_x, feature_names, preset['powers'])
    test_x = add_powers(test_x, feature_names, preset['powers'])[0]

maes = []

for split in range(n_splits):
    print
    print("Training split %d..." % split)

    for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=2016 + 17*split)):
        if args.fold is not None and fold != args.fold:
            continue

        print
        print("  Fold %d..." % fold)

        fold_train_x = train_x[fold_train_idx]
        fold_train_y = train_y[fold_train_idx]
        fold_train_r = train_r.slice(fold_train_idx)

        fold_eval_x = train_x[fold_eval_idx]
        fold_eval_y = train_y[fold_eval_idx]
        fold_eval_r = train_r.slice(fold_eval_idx)

        fold_test_x = test_x
        fold_test_r = test_r

        fold_feature_names = list(feature_names)

        if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
            print("    Building per-fold features...")

            fold_train_x = [fold_train_x]
            fold_eval_x = [fold_eval_x]
            fold_test_x = [fold_test_x]

            for fb in feature_builders:
                fold_train_x.append(fb.fit_transform(fold_train_r))
                fold_eval_x.append(fb.transform(fold_eval_r))
                fold_test_x.append(fb.transform(fold_test_r))
                fold_feature_names += fb.get_feature_names()

            fold_train_x = hstack(fold_train_x)
            fold_eval_x = hstack(fold_eval_x)
            fold_test_x = hstack(fold_test_x)

        eval_p = np.zeros((fold_eval_x.shape[0], n_bags))

        for bag in range(n_bags):
            print("    Training model %d..." % bag)

            rs = np.random.RandomState(101 + 31*split + 13*fold + 29*bag)

            bag_train_x = fold_train_x
            bag_train_y = fold_train_y

            bag_eval_x = fold_eval_x
            bag_eval_y = fold_eval_y

            bag_test_x = fold_test_x

            if 'sample' in preset:
                bag_train_x, bag_train_y = resample(fold_train_x, fold_train_y, replace=False, n_samples=int(preset['sample'] * fold_train_x.shape[0]), random_state=42 + 11*split + 13*fold + 17*bag)

            if 'feature_sample' in preset:
                features = rs.choice(range(bag_train_x.shape[1]), int(bag_train_x.shape[1] * preset['feature_sample']), replace=False)

                bag_train_x = bag_train_x[:, features]
                bag_eval_x = bag_eval_x[:, features]
                bag_test_x = bag_test_x[:, features]

            if 'svd' in preset:
                svd = TruncatedSVD(preset['svd'])

                bag_train_x = svd.fit_transform(bag_train_x)
                bag_eval_x = svd.transform(bag_eval_x)
                bag_test_x = svd.transform(bag_test_x)
            pe, pt = preset['model'].fit_predict(train=(bag_train_x, y_transform(bag_train_y)),
                                                 val=(bag_eval_x, y_transform(bag_eval_y)),
                                                 test=(bag_test_x, ),
                                                 seed=42 + 11*split + 17*fold + 13*bag,
                                                 feature_names=fold_feature_names,
                                                 eval_func=lambda yt, yp: accuracy_score(y_inv_transform(yt), y_inv_transform(yp)),
                                                 name='%s-fold-%d-%d' % (args.preset, fold, bag))

            eval_p[:, bag] += pe
            test_foldavg_p[:, split * n_folds * n_bags + fold * n_bags + bag] = pt

            train_p[fold_eval_idx, split * n_bags + bag] = pe

            print("    Accuracy of model: %.5f" % accuracy_score(fold_eval_y, y_inv_transform(pe)))

        print("  Accuracy of mean-transform: %.5f" % accuracy_score(fold_eval_y, y_inv_transform(y_aggregator(eval_p, axis=1))))
        print("  Accuracy of transform-mean: %.5f" % accuracy_score(fold_eval_y, y_aggregator(y_inv_transform(eval_p), axis=1)))
        print("  Accuracy of transform-median: %.5f" % accuracy_score(fold_eval_y, y_aggregator(y_inv_transform(eval_p), axis=1)))

        # Calculate err
        maes.append(accuracy_score(fold_eval_y, y_aggregator(y_inv_transform(eval_p), axis=1)))

        print("  Accuracy: %.5f" % maes[-1])

        # Free mem
        del fold_train_x, fold_train_y, fold_eval_x, fold_eval_y


if True:
    print
    print("  Full...")

    full_train_x = train_x
    full_train_y = train_y
    full_train_r = train_r

    full_test_x = test_x
    full_test_r = test_r

    full_feature_names = list(feature_names)

    if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
        print("    Building per-fold features...")

        full_train_x = [full_train_x]
        full_test_x = [full_test_x]

        for fb in feature_builders:
            full_train_x.append(fb.fit_transform(full_train_r))
            full_test_x.append(fb.transform(full_test_r))
            full_feature_names += fb.get_feature_names()

        full_train_x = hstack(full_train_x)
        full_test_x = hstack(full_test_x)

    for bag in range(n_bags):
        print("    Training model %d..." % bag)

        rs = np.random.RandomState(101 + 31*split + 13*fold + 29*bag)

        bag_train_x = full_train_x
        bag_train_y = full_train_y

        bag_test_x = full_test_x

        if 'sample' in preset:
            bag_train_x, bag_train_y = resample(bag_train_x, bag_train_y, replace=False, n_samples=int(preset['sample'] * bag_train_x.shape[0]), random_state=42 + 11*split + 13*fold + 17*bag)

        if 'feature_sample' in preset:
            features = rs.choice(range(bag_train_x.shape[1]), int(bag_train_x.shape[1] * preset['feature_sample']), replace=False)

            bag_train_x = bag_train_x[:, features]
            bag_test_x = bag_test_x[:, features]

        if 'svd' in preset:
            svd = TruncatedSVD(preset['svd'])

            bag_train_x = svd.fit_transform(bag_train_x)
            bag_test_x = svd.transform(bag_test_x)

        pt = preset['model'].fit_predict(train=(bag_train_x, y_transform(bag_train_y)),
                                         test=(bag_test_x, ),
                                         seed=42 + 11*split + 17*fold + 13*bag,
                                         feature_names=fold_feature_names,
                                         eval_func=lambda yt, yp: accuracy_score(y_inv_transform(yt), y_inv_transform(yp)),
                                         size_mult=n_folds / (n_folds - 1.0),
                                         name='%s-full-%d' % (args.preset, bag))

        test_fulltrain_p[:, bag] = pt


# Aggregate predictions
#print(y_aggregator(y_inv_transform(train_p), axis=1).unstack())
train_p = pd.Series(y_aggregator(y_inv_transform(train_p), axis=1), index=Dataset.load_part('train', 'id'))
test_foldavg_p = pd.Series(y_aggregator(y_inv_transform(test_foldavg_p), axis=1), index=Dataset.load_part('test', 'id'))
test_fulltrain_p = pd.Series(y_aggregator(y_inv_transform(test_fulltrain_p), axis=1), index=Dataset.load_part('test', 'id'))

# Analyze predictions
mae_mean = np.mean(maes)
mae_std = np.std(maes)
cv_score = accuracy_score(train_y, train_p)

print
print("CV Accuracy: %.5f +- %.5f" % (mae_mean, mae_std))
print("CV RES Accuracy: %.5f" % cv_score)

name = "%s-%s-%.5f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), args.preset, cv_score)

print()
print("Saving predictions... (%s)" % name)
print()
print()

for part, pred in [('train', train_p), ('test-foldavg', test_foldavg_p), ('test-fulltrain', test_fulltrain_p)]:
    pred.rename('Loan_Status', inplace=True)
    pred.index.rename('Loan_ID', inplace=True)
    pred.to_csv('preds/%s-%s.csv' % (name, part), header=True)

copy2(os.path.realpath(__file__), os.path.join("preds", "%s-code.py" % name))

df = pd.DataFrame(columns=["model","CV"])
df.loc[len(df)] = ["'" + name + "',", cv_score]
import os
# if file does not exist write header
if not os.path.isfile('results/models.csv'):
   df.to_csv('results/models.csv',index_label=None)
else: # else it exists so append without writing the header
    df.to_csv('results/models.csv',mode = 'a',header=False,index_label=None)

print("Done.")
