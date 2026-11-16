import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))
import config
import classifiers
import labels
import joblib
import pandas as pd
import numpy as np

model_path = config.PATHS['classifiers_folder'] / (config.RF_TRAINING['model_name'] + '_best_estimator' + '.pkl')
model = joblib.load(model_path)

test_features_folder = config.PATHS['features_folder'] / 'test'
test_labels_folder = config.PATHS['labels_folder'] / 'test'

test_features = sorted(list(test_features_folder.glob('*.csv')))
test_labels = sorted(list(test_labels_folder.glob('*.csv')))

if not test_features:
    raise IOError('No test features found in the test folder')
if not test_labels:
    raise IOError('No test labels found in the test folder')

all_features = [pd.read_csv(feature, index_col=0, header=0) for feature in test_features]
all_labels = [labels.convert_to_binary(label, behavior=config.BEHAVIOR, slice=config.RF_TRAINING['label_slice']) for label in test_labels]

x = pd.concat(all_features)
y = np.concatenate(all_labels)

classifiers.classification_report(model,
                                  x,
                                  y,
                                  save=True,
                                  save_path=(config.PATHS['classifiers_folder'] / (config.RF_TRAINING['model_name'] + '_test_report.png')),
                                  negative_class=f'Not {config.BEHAVIOR.replace("_", " ")}',
                                  positive_class=config.BEHAVIOR.replace('_', ' ').capitalize())
