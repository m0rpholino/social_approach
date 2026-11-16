# Save behavior and distance summaries to Excel files.
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))
import config
import behavior
import pandas as pd
import joblib

tracks = list(config.PATHS['tracks_folder'].glob('*.h5'))
features = list(config.FEATURE_EXTRACTION['features_extracted_path'].glob('*.csv')) # what if they're split in train and test?
model_path = config.PATHS['classifiers_folder'] / (config.RF_TRAINING['model_name'] + '_best_estimator.pkl')
if not model_path.is_file():
    raise IOError(f"Model {model_path} not found")
model = joblib.load(model_path)

distance_analysis = True
behavior_analysis = True
if not tracks:
    print('No tracking files found in the tracks folder. Skipping distance traveled analysis')
    distance_analysis = False
if not features:
    print('No features found in the features_extracted folder in the simba project. Skipping behavioral analysis')
    behavior_analysis = False
    
# configured behavior
print(f"Calculating {config.BEHAVIOR} with analysis type {config.ANALYSIS['behavior_type']}")
save_folder = config.PATHS['prediction_folder']
if behavior_analysis:
    if config.ANALYSIS['behavior_type'] == 'total':
        
        out_df = behavior.total_behavior(
            model=model,
            features_path=features,
            min_bout_duration_ms=config.RF_INFERENCE['min_bout_duration_ms'],
            fps=config.VIDEOS['fps'],
            threshold=config.RF_INFERENCE['threshold'],
        )
        
        out_filepath = Path(save_folder) / f'total_{config.BEHAVIOR}.xlsx'
        with pd.ExcelWriter(out_filepath) as writer:
            out_df.to_excel(writer)


    elif config.ANALYSIS['behavior_type'] == 'bins':
        if not isinstance(config.ANALYSIS['bin_duration_s'], int):
            raise ValueError(f"Unsupported type of bin duration: {config.ANALYSIS['bin_duration_s']}")

        out_df = behavior.bin_behavior(
            model=model,
            features_path=features,
            min_bout_duration_ms=config.RF_INFERENCE['min_bout_duration_ms'],
            fps=config.VIDEOS['fps'],
            threshold=config.RF_INFERENCE['threshold'],
            bin_duration_s=config.ANALYSIS['bin_duration_s'],
        )
        out_filepath = Path(save_folder) / f'bins_{config.BEHAVIOR}.xlsx'

        with pd.ExcelWriter(out_filepath) as writer:
            out_df.to_excel(writer)


    elif config.ANALYSIS['behavior_type'] == 'bouts':
        out_df = behavior.bout_behavior(
            model=model,
            features_path=features,
            min_bout_duration_ms=config.RF_INFERENCE['min_bout_duration_ms'],
            fps=config.VIDEOS['fps'],
            threshold=config.RF_INFERENCE['threshold'],
        )
        filepath = Path(save_folder) / f'bouts_{config.BEHAVIOR}.xlsx'
        with pd.ExcelWriter(filepath) as writer:
            out_df.to_excel(writer)

    elif config.ANALYSIS['behavior_type'] == 'bin_bouts':
        if not isinstance(config.ANALYSIS['bin_duration_s'], int):
            raise ValueError(f"Unsupported type of bin duration: {config.ANALYSIS['bin_duration_s']}")
        out_df = behavior.bout_bin_behavior(
            model=model,
            features_path=features,
            min_bout_duration_ms=config.RF_INFERENCE['min_bout_duration_ms'],
            fps=config.VIDEOS['fps'],
            threshold=config.RF_INFERENCE['threshold'],
            bin_duration_s=config.ANALYSIS['bin_duration_s'],
        )
        filepath = Path(save_folder) / f'bins_bouts_{config.BEHAVIOR}.xlsx'
        with pd.ExcelWriter(filepath) as writer:
            out_df.to_excel(writer)
    elif config.ANALYSIS['behavior_type'] is None:
        pass

    else:
        raise ValueError(f"Unsupported type of behavior analysis: {config.ANALYSIS['behavior_type']}")

# distance traveled
if distance_analysis:
    if config.ANALYSIS['distance_type'] == 'total':
        dist_df = behavior.total_distance(config.PATHS['tracks_folder'])
        dist_filepath = Path(save_folder) / 'total_distance.xlsx'
        with pd.ExcelWriter(dist_filepath) as writer:
            dist_df.to_excel(writer)

    elif config.ANALYSIS['distance_type'] == 'bins':
        dist_df = behavior.bin_distance(config.PATHS['tracks_folder'],
                                        config.ANALYSIS['bin_duration_s'],
                                        config.VIDEOS['fps'])
        dist_filepath = Path(save_folder) / f'bins_distance.xlsx'
        with pd.ExcelWriter(dist_filepath) as writer:
            dist_df.to_excel(writer)

    elif config.ANALYSIS['distance_type'] is None:
        pass

    else:
        raise ValueError(f"Unsupported type of distance analysis: {config.ANALYSIS['distance_type']}")
