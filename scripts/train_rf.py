import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

import config
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import labels
import numpy as np
import classifiers

features_paths = sorted(list(config.PATHS['features_folder'].glob('*.csv')))
label_paths = sorted(list(config.PATHS['labels_folder'].glob('*.csv')))

# split data in test and train
classifiers.train_test_split(config.PATHS['labels_folder'],
                             config.PATHS['features_folder'],
                             config.RF_TRAINING['training_fraction'])

train_features_paths = sorted(list((config.PATHS['features_folder'] / 'train').glob('*.csv')))
train_label_paths = sorted(list((config.PATHS['labels_folder'] / 'train').glob('*.csv')))

all_features = []
all_labels = []
all_groups = []

for path in train_features_paths:
    feature = pd.read_csv(path, index_col=0, header=0)
    print(f'Len of {path.stem}: {len(feature)}')
    video = path.stem
    all_groups.extend([video]*len(feature))
    all_features.append(feature)
x = pd.concat(all_features, ignore_index=True)

for path in train_label_paths:
    binary_labels = labels.convert_to_binary(path,
                                      behavior=config.BEHAVIOR,
                                      slice=config.RF_TRAINING['label_slice'])
    all_labels.append(binary_labels)
y = np.concatenate(all_labels)


grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42,
                                                            n_jobs=1,
                                                            criterion=config.RF_TRAINING['criterion']),
                                                            param_grid = config.RF_TRAINING['param_grid'],
                                                            cv=StratifiedGroupKFold(n_splits=config.RF_TRAINING['n_splits']),
                                                            scoring=config.RF_TRAINING['scoring'],
                                                            refit=config.RF_TRAINING['refit'],
                                                            n_jobs=config.RF_TRAINING['n_jobs'],
                                                            verbose=1)
result = grid_search.fit(x,
                y,
                groups=all_groups)

print(f'Best parameters: {result.best_params_}')
print(f'Best score: {result.best_score_}')

best_idx = result.best_index_
print("\nAll metrics for best model:")
for metric_name in  config.RF_TRAINING['scoring'].keys():
    mean = result.cv_results_[f'mean_test_{metric_name}'][best_idx]
    std = result.cv_results_[f'std_test_{metric_name}'][best_idx]
    print(f"{metric_name}: {mean:.3f} (+/- {std:.3f})")

joblib.dump(result, (config.PATHS['classifiers_folder'] / (config.RF_TRAINING['model_name'] + '_grid_search' +'.pkl')))
joblib.dump(result.best_estimator_, (config.PATHS['classifiers_folder'] / (config.RF_TRAINING['model_name'] + '_best_estimator' +'.pkl')))

classifiers.cv_report(grid_search=(config.PATHS['classifiers_folder'] / (config.RF_TRAINING['model_name'] + '_grid_search' +'.pkl')),
                      classes=[f'not_{config.BEHAVIOR}', config.BEHAVIOR],
                      save=True,
                      save_path=(config.PATHS['classifiers_folder'] / (config.RF_TRAINING['model_name'] + '_cv_report.png')))