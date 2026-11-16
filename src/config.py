from pathlib import Path
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(r'/path/to/your/folder') # path to the project folder

BEHAVIOR = 'social_approach'

PATHS = {
    'videos_folder':PROJECT_ROOT / 'videos',
    'excluded_videos_folder': PROJECT_ROOT /'videos' / 'test_animal',
    'crop_coordinates_folder': PROJECT_ROOT / 'crop_coordinates',
    'tracks_folder':PROJECT_ROOT / 'tracks',
    'labels_folder':PROJECT_ROOT / 'labels',
    'feature_extraction_folder':PROJECT_ROOT / 'feature_extraction',
    'features_folder':PROJECT_ROOT / 'feature_extraction' / 'project_folder' / 'csv' / 'features_extracted',
    'prediction_folder':PROJECT_ROOT / 'predictions',
    'classifiers_folder':PROJECT_ROOT / 'classifiers',
    'trackers_folder':PROJECT_ROOT / 'trackers'
}

VIDEOS = {
    'format': 'mp4', # video file extension
    'cages': 2, # number of cages in a video
    'fps': 30, # frames per second
    'wall_length_mm': 600, # length of the divider wall, assumed to match the height of the video
    'test_animal_side': 'left', # side of the cage where the test animal is (either 'left' or 'right')
    'crop_example_videos':['example_video_1',
                           'example_video_2'
                           ] # list of videos to use to get crop coordinates.
}

DLC = {
    'config_path': r'your/path/to/dlc/config' # path to a deeplabcut configuration file (.yaml)
}

TRACKS = {
    'slice': (), # slice of tracks to process for feature extraction in frames. To process the whole tracks, set to ()
    'p_cutoff': 0.9, # likelihood cutoff to apply before outlier correction for feature extraction and distance calculation
    'reference_bodyparts': ('nose', 'tailbase'), # bodyparts to use as reference distance for outlier correction
    'movement_criterion': 1.5, # criterion to find movement outliers. Frames that move more than criterion*reference between consecutive frames are flagged
    'location_criterion': 2.0, # criterion to find location outliers. Frames more distant than criterion*reference to at least 2 bodyparts are flagged
    'max_interpolation': 5, # maximum consecutively missing frames to interpolate. If None, all frames below likelihood threshold will be interpolated
    'discard_animal_absent': True # if True, frames at the start where less than half bodyparts are above 0.9 likelihood are discarded
    }

RF_TRAINING = {
    'model_name':'classifier1',
    'n_splits':5, # number of folds to split the data into for cross validation
    'criterion':'entropy', # loss function for random forest training. Can be 'entropy', 'gini', or 'log-loss'
    'training_fraction': 0.9, # fraction of videos to use for the training set. The rest is assigned to the test set
    'scoring':{ # metrics to use for cross validation.
    'f1_macro': 'f1_macro',
    f'f1_not_{BEHAVIOR}': make_scorer(f1_score, pos_label=0, average='binary'),
    f'f1_{BEHAVIOR}': make_scorer(f1_score, pos_label=1, average='binary'),
    f'precision_not_{BEHAVIOR}': make_scorer(precision_score, pos_label=0, average='binary'),
    f'precision_{BEHAVIOR}': make_scorer(precision_score, pos_label=1, average='binary'),
    f'recall_not_{BEHAVIOR}': make_scorer(recall_score, pos_label=0, average='binary'),
    f'recall_{BEHAVIOR}': make_scorer(recall_score, pos_label=1, average='binary'),
    },
    'param_grid':{ # hyperparameter grid for random forest training. Grid search will evaluate all combinations of parameters
        'min_samples_leaf': [10, 20], # minimum number of samples at the final leaf
              'n_estimators': [100, 200], # number of decision trees
                'max_depth': [10, 15] # maximum depth of decision trees
                },

    'refit':'f1_macro', # metric to maximize during cross validation. Must match a metric defined in scoring
    'n_jobs': 1, # number of CPU cores to use for random forest training
    'label_slice': () # duration of labeling videos in (start_frame, stop_frame) format
}

RF_INFERENCE = {
    'min_bout_duration_ms': 2000, # minimum bout duration in milliseconds
    'threshold': 0.5 # threshold to count a raw prediction as behavior
}

FEATURE_EXTRACTION = {
	'project_name': "feature_extraction",
	'clf_name': ["seeking"],
	'bp_count': "7",
	'bp_config_idx': 1,
	'animal_count': 1,
	'filetype': "csv",
	'subset': [
		"FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)",
		"FRAME-BY-FRAME BODY-PARTS INSIDE ROIS (BOOLEAN)",
		"BODY-PART DISTANCES TO VIDEO FRAME EDGE (MM)",
		"WITHIN-ANIMAL THREE-POINT BODY-PART ANGLES (DEGREES)"
	]
}

FEATURE_EXTRACTION['project_folder_path'] = PROJECT_ROOT / FEATURE_EXTRACTION['project_name'] / 'project_folder'
FEATURE_EXTRACTION['project_config_path'] = FEATURE_EXTRACTION['project_folder_path'] / 'project_config.ini'
FEATURE_EXTRACTION['processed_tracks_path'] = FEATURE_EXTRACTION['project_folder_path'] / 'csv' / 'outlier_corrected_movement_location'
FEATURE_EXTRACTION['features_extracted_path'] = FEATURE_EXTRACTION['project_folder_path'] / 'csv' / 'features_extracted'

ANALYSIS = {'behavior_type':'total', # can be 'total', 'bins', 'bouts', 'bin_bouts' or None to skip
            'distance_type': 'total', # can be 'total' or 'bins' or None to skip
            'bin_duration_s':300} 
