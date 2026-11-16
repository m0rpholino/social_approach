from pathlib import Path
from typing import List, Tuple, Optional, Union
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import shutil

def precision(predictions: np.ndarray,
                labels: np.ndarray) -> np.ndarray:
    """
    Compute precision for each class in a binary classification setting.
    Params:
        predictions (np.ndarray): binary classifier predictions (0 or 1)
        labels (pd.Series): binary ground truth (0 or 1)
    Returns:
        np.ndarray: [precision_negative, precision_positive]
    """
    predictions = predictions.astype(bool)
    labels = labels.astype(bool)

    tp = (predictions & labels).sum()
    fp = (predictions & ~labels).sum()
    tn = (~predictions & ~labels).sum()
    fn = (~predictions & labels).sum()

    precision_positive = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    precision_negative = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return np.array([precision_negative, precision_positive])

def recall(predictions: np.ndarray,
            labels: np.ndarray) -> np.ndarray:
    """
    Compute recall for each class in a binary classification setting.
    Params:
        predictions (np.ndarray): binary classifier predictions (0 or 1)
        labels (pd.Series): binary ground truth (0 or 1)
    Returns:
        np.ndarray: [recall_negative, recall_positive]
    """
    predictions = predictions.astype(bool)
    labels = labels.astype(bool)

    tp = (predictions & labels).sum()
    fp = (predictions & ~labels).sum()
    tn = (~predictions & ~labels).sum()
    fn = (~predictions & labels).sum()

    recall_positive = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_negative = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return np.array([recall_negative, recall_positive])

def f1(precision_score: np.ndarray,
        recall_score: np.ndarray) -> np.ndarray:
    """
    Compute f1 score from precision and recall arrays.
    Params:
        precision (np.ndarray): [precision_negative, precision_positive]
        recall (np.ndarray): [recall_negative, recall_positive]
    Returns:
        np.ndarray: [f1_negative, f1_positive]
    """
    denom = precision_score + recall_score
    denom = np.where(denom == 0, 1, denom)  # avoid division by zero
    return (2 * precision_score * recall_score) / denom

def cv_report(grid_search: Union[str, Path, GridSearchCV],
                classes = List[str],
                save: Optional[bool] = False,
                save_path: Optional[bool] = (Path.cwd() / 'cv_report.png')):
    """
    Generate a report table with precision, recall, and F1 per class
    Params:
        grid_search (str, Path, GridSearchCV): trained GridSearchCV object
        classes (List[str]): list of class names. Must match class names in the GridSearchCV object. Order determines order in the table
        save (Optional, bool): if True, save the figure as a png
        save_path (Optional, str, Path): full path to save the file into. If not specified, will be saved in working directory
    Returns:
        None
    """
    # load grid search model
    if isinstance(grid_search, str) or isinstance(grid_search, Path):
        grid_search = joblib.load(grid_search)
    elif isinstance(grid_search, GridSearchCV):
        grid_search = grid_search
    else:
        raise TypeError(f'grid_search should be str, path or GridSearchCV, got {type(grid_search)} instead')
    
    # Organize data: [metric][class] = (mean, std)
    metrics = ['precision', 'recall', 'f1']
    best_idx = grid_search.best_index_
    
    data = pd.DataFrame(index=metrics)

    for class_name in classes:
        for metric in metrics:
            mean_metric = grid_search.cv_results_[f'mean_test_{metric}_{class_name}'][best_idx]
            std_metric = grid_search.cv_results_[f'std_test_{metric}_{class_name}'][best_idx]
            data.loc[metric, f'mean_{metric}_{class_name}'] = mean_metric
            data.loc[metric, f'std_{metric}_{class_name}'] = std_metric

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 5))

    # Create colormap (red to green)
    cmap = LinearSegmentedColormap.from_list('crimson_teal', ['#c0392b' , '#16a085'])
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Create table data and colors
    cell_text = []
    cell_colors = []

    for metric in metrics:
        row_text = []
        row_colors = []
        for class_name in classes:
            mean, std = data.loc[metric, f'mean_{metric}_{class_name}'], data.loc[metric, f'std_{metric}_{class_name}']
            row_text.append(f'{mean:.3f} ± {std:.3f}')
            # Color based on mean value
            color = cmap(norm(mean))
            row_colors.append(color)
        cell_text.append(row_text)
        cell_colors.append(row_colors)
    
    # stylize labels
    styled_metrics = [metric.capitalize() for metric in metrics]
    styled_classes = [class_name.replace('_', ' ').capitalize() if '_' in class_name else class_name.capitalize() for class_name in classes]

    # Create the table
    table = ax.table(cellText=cell_text,
                    colLabels=styled_classes,
                    rowLabels=styled_metrics,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.35, 0.35],
                    cellColours=None,
                    rowColours=None)

    # Set row height
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Apply colors to cells
    for i in range(len(metrics)):
        for j in range(len(classes)):
            cell = table[(i+1, j)]
            cell.set_facecolor(cell_colors[i][j])
            cell.set_text_props(weight='bold', fontsize=28, va='center')

    # Style header and row labels
    for i in range(len(classes)):
        table[(0, i)].set_facecolor('#cccccc')
        table[(0, i)].set_text_props(weight='bold', fontsize=32)

    for i in range(len(metrics)):
        table[(i+1, -1)].set_facecolor('#cccccc')
        table[(i+1, -1)].set_text_props(weight='bold', fontsize=32)

    ax.axis('off')
    plt.title('5-Fold Cross Validation Results', fontsize=42)
    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
    plt.show()

def test(model: RandomForestClassifier,
            test_features: pd.DataFrame,
            test_labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a fitted RandomForestClassifier on test data.
    Params:
        model (RandomForestClassifier): fitted random forest classifier
        test_features (pd.DataFrame): features to predict on
        test_labels (pd.Series): binary ground truth (0 or 1)
    Returns:
        precision, recall, f1 (Tuple[np.ndarray]): scores for negative and positive class
    """
    raw_predictions = model.predict_proba(test_features)
    binary_predictions = np.argmax(raw_predictions, axis=1)

    precision_score = precision(binary_predictions, test_labels)
    recall_score = recall(binary_predictions, test_labels)
    f1_score = f1(precision_score, recall_score)

    return precision_score, recall_score, f1_score

def classification_report(model: RandomForestClassifier,
                            test_features: pd.DataFrame,
                            test_labels: pd.Series,
                            save: bool = False,
                            save_path: Optional[Union[str, Path]] = None,
                            negative_class: Optional[str] = None,
                            positive_class: Optional[str] = None) -> None:
    """
    Generate and display a color-coded classification report table.
    Params:
        model (RandomForestClassifier): fitted random forest classifier
        test_features (pd.DataFrame): features to predict on
        test_labels (pd.Series): binary ground truth (0 or 1)
        save (bool): whether to save the figure. Default is False
        save_path (Optional[Union[str, Path]]): path to save the figure. Used only if save is True
        negative_class (Optional[str]): label for the negative class column. Default is 'Negative'
        positive_class (Optional[str]): label for the positive class column. Default is 'Positive'
    Returns:
        None
    """
    precision_score, recall_score, f1_score = test(model, test_features, test_labels)

    cell_data = np.array([precision_score, recall_score, f1_score])  # shape (3, 2): rows=metrics, cols=classes
    row_labels = ['Precision', 'Recall', 'F1']
    col_labels = [negative_class if negative_class is not None else 'Negative',
                    positive_class if positive_class is not None else 'Positive']

    cmap = LinearSegmentedColormap.from_list('crimson_teal', ['#c0392b' , '#16a085'])
    norm = mcolors.Normalize(vmin=0, vmax=1)

    cell_text = [[f'{v:.3f}' for v in row] for row in cell_data]
    cell_colors = [[cmap(norm(v)) for v in row] for row in cell_data]

    # scale figure width to the longest text in each column and row labels
    char_width = 0.13  # approximate inches per character at fontsize 12
    col_widths = [max(len(col_labels[c]), max(len(cell_text[r][c]) for r in range(len(row_labels)))) for c in range(len(col_labels))]
    row_label_width = max(len(r) for r in row_labels)
    fig_width = (sum(col_widths) + row_label_width) * char_width + 0.5
    fig_height = (len(row_labels) + 1) * 0.45

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    table = ax.table(cellText=cell_text,
                        cellColours=cell_colors,
                        rowLabels=row_labels,
                        cellLoc='center',
                        loc='center',
                        colLabels=col_labels,
                        colWidths = [0.35, 0.35],
                        rowColours=None)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(cell_data.shape[0]):
        for j in range(cell_data.shape[1]):
            cell = table[(i+1, j)]
            cell.set_facecolor(cell_colors[i][j])
            cell.set_text_props(weight='bold', fontsize=28, va='center')

    # Style header and row labels
    for i in range(cell_data.shape[1]): # columns
        table[(0, i)].set_facecolor('#cccccc')
        table[(0, i)].set_text_props(weight='bold', fontsize=32)

    for i in range(cell_data.shape[0]): # rows
        table[(i+1, -1)].set_facecolor('#cccccc')
        table[(i+1, -1)].set_text_props(weight='bold', fontsize=32)

    plt.tight_layout()
    plt.title('Test set results')

    if save and save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

    plt.show()

def train_test_split(label_folder: Union[str, Path],
                     features_folder: Union[str, Path],
                     train_fraction: float) -> None:
    """
    Split features and labels into train and test sets.
    Params:
        label_folder (str, Path): path to a folder containing only label files
        features_folder (str, Path): path to a folder containing only feature files
        train_fraction (float): fraction of data to assign to training test
    Returns:
        None
    """
    # check label_folder type and convert to Path
    if isinstance(label_folder, str):
        label_folder = Path(label_folder)
    elif not isinstance(label_folder, Path):
        raise TypeError(f'Unsupported type for label_folder: {type(label_folder)}')
    
    #check features_folder type and convert to Path
    if isinstance(features_folder, str):
        features_folder = Path(features_folder)
    elif not isinstance(features_folder, Path):
        raise TypeError(f'Unsupported type for features_folder: {type(features_folder)}')
    
    # check if train and test folders exist
    label_train_folder = label_folder / 'train'
    label_test_folder = label_folder / 'test'

    feature_train_folder = features_folder / 'train'
    feature_test_folder = features_folder / 'test'

    skip = False
    first_split = True
    all_folders = [label_train_folder, label_test_folder, feature_train_folder, feature_test_folder]
    if all(folder.is_dir() for folder in all_folders):
        if all(len(list(folder.iterdir())) != 0 for folder in all_folders):
            response = input('A train-test split as already been found. Generate a new one? [y/n]')
            if response == 'n':
                first_split = False
                skip = True
            elif response == 'y':
                skip = False
                first_split = False
            else:
                raise ValueError('Invalid answer')
    
    if skip:
        return None
    if not first_split:
        for folder in all_folders:
            files = list(folder.glob('*.csv'))
            for file in files:
                shutil.move(file, folder.parent)
        # retrieve sorted lists of labels and features - assumes same name
        labels = sorted(list(label_folder.glob('*.csv')))
        features = sorted(list(features_folder.glob('*.csv')))
    else:
        labels = sorted(list(label_folder.glob('*.csv')))
        features = sorted(list(features_folder.glob('*.csv')))

    # check if labels and features are present
    if len(labels) == 0:
        raise ValueError('No labels found in folder')
    if len(features) == 0:
        raise ValueError('No features found in folder')
    if len(labels) != len(features):
        raise ValueError('Number of labels and features files must match')
    
    # combine labels and features in tuples and shuffle
    combined = [(label, feature) for label, feature in zip(labels, features)]
    random.shuffle(combined)

    # retrieve the index to perform split at
    split_index = round(len(labels) * train_fraction)
    train_set = combined[:split_index]
    test_set = combined[split_index:]

    # create test and training folders
    label_train_folder = label_folder / 'train'
    label_test_folder = label_folder / 'test'

    feature_train_folder = features_folder / 'train'
    feature_test_folder = features_folder / 'test'

    label_train_folder.mkdir(exist_ok=True)
    label_test_folder.mkdir(exist_ok=True)

    feature_train_folder.mkdir(exist_ok=True)
    feature_test_folder.mkdir(exist_ok=True)
    
    # move splits to relevant folders
    for train_data in train_set:
        shutil.move(train_data[0], label_train_folder)
        shutil.move(train_data[1], feature_train_folder)
    
    for test_data in test_set:
        shutil.move(test_data[0], label_test_folder)
        shutil.move(test_data[1], feature_test_folder)
