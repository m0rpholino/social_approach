from pathlib import Path
from numbers import Real
from typing import List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import config
import tracksprocessor as tp
import videoprocessor as vp
import re

def run_inference(model: Union[str, Path, RandomForestClassifier],
                  features_path: Union[str, Path],
                  save_folder: Optional[Union[str, Path]] = None) -> np.ndarray:
    """
    Get raw predictions from a list of features.
    Params:
        model (str | path | RandomForestClassifier): trained RandomForestClassifier object or path to it
        features (List[str] | List[path]): list of paths to features in csv format
        save_folder (Optional, str | Path): folder to save predictions to
    Return:
        all_preds (Dict[str, np.ndarray]): dictionary of raw predictions per features file name
    """
    if isinstance(model, str) or isinstance(model, Path):
            model = joblib.load(model)
    elif isinstance(model, RandomForestClassifier):
        model = model
    else:
        raise TypeError(f'Unsupported type for model: {type(model)}')
        
    if save_folder is not None:
        if isinstance(save_folder, str):
            save_folder = Path(save_folder)

    if isinstance(features_path, str):
        features_path = Path(features_path)
    feature_df = pd.read_csv(features_path, index_col=0)
    preds = model.predict_proba(feature_df)
    if save_folder:
        save_path = save_folder / (features_path.stem + '.npy')
        np.save(str(save_path), preds)

    return preds

def binary_preds(preds: np.ndarray,
                threshold: Optional[float] = config.RF_INFERENCE['threshold']) -> np.ndarray:
    """
    Round raw predictions for behavior according to a threshold.
    Params:
        threshold (float): minimum probability to classify frame as behavior
    Returns:
        rounded_preds (np.ndarray): rounded predictions
    """
    return (preds[:, 1] >= threshold).astype(int)

def bin_preds(preds: np.ndarray,
         bin_seconds: int,
         fps: Optional[int] = 30) -> List[np.ndarray]:
    bin_frames = bin_seconds*fps
    bin_arrays = [preds[i:i+bin_frames] for i in range(0, len(preds), bin_frames)]
    return bin_arrays

def bout_boundaries(binary_preds: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[Tuple[np.ndarray]]]:
    """
    Retrieve an array with the length of each bout
    Params:
        binary_preds (np.ndarray): binary predictions. It's advisable to use predictions filtered by minimum bout lengths
    Returns:
        bouts (Union[np.ndarray, List[np.ndarray]): array of length (in frames) of each bout
    """
    if isinstance(binary_preds, np.ndarray):
        pad_row = [0]
        padded = np.r_[pad_row, binary_preds, pad_row]
        
        diff = np.diff(padded, axis=-1)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        return starts, ends

    elif isinstance(binary_preds, list): # predictions binned
        bout_boundaries = []
        for array in binary_preds:
            pad_row = [0]
            padded = np.r_[pad_row, array, pad_row]
            
            diff = np.diff(padded, axis=-1)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            bout_boundaries.append((starts, ends))
        return bout_boundaries # list of (starts, ends) tuples for each bin

def filter_bouts(binary_preds: np.ndarray,
                 min_bout_duration_ms: int,
                 fps: int) -> np.ndarray:
    """
    Filter a binary series to keep only behavior bouts above minimum duration.
    Params:
        binary_preds (np.ndarray): array of binary predictions
        min_bout_duration_ms (int): minimum bout duration in milliseconds
        fps (int): video frame rate

    Returns:
        filtered (np.ndarray): binary predictions with short positive bouts removed
    """
    if not isinstance(binary_preds, np.ndarray):
        raise TypeError(f'binary_preds should be np.ndarray or list of np.ndarray, found {type(binary_preds)} instead')

    if isinstance(min_bout_duration_ms, bool) or not isinstance(min_bout_duration_ms, Real):
        raise TypeError('min_bout_duration_ms must be a number')
    if min_bout_duration_ms < 0:
        raise ValueError('min_bout_duration_ms must be greater than or equal to 0')
    if isinstance(fps, bool) or not isinstance(fps, Real):
        raise TypeError('fps must be a number')
    if fps <= 0:
        raise ValueError('fps must be greater than 0')

    filtered = np.zeros_like(binary_preds, dtype=int)
    starts, ends = bout_boundaries(binary_preds)
    lengths = ends - starts

    keep = lengths * 1000 >= min_bout_duration_ms * fps
    for start, end, keep_flag in zip(starts, ends, keep):
        if keep_flag:
            filtered[start:end] = 1

    return filtered
    
def preds_to_percent(pred_array: Union[np.ndarray, List[np.ndarray]],
                  fps: Optional[int] = 30) -> Union[float, List[float]]:
    """
    Convert binary predictions into percent of time of positive class.
    Params:
        pred_array (np.ndarray, List[np.ndarray]): binary prediction array
        fps (Optional, int): frames per seconds of the original video
    Returns:
        percent_behavior (float, List[float]): percent of total behavior or per bin
    """
    if isinstance(pred_array, np.ndarray):
        behavior_frames = pred_array.sum()
        percent_behavior = (behavior_frames / len(pred_array))*100
        return percent_behavior
    elif isinstance(pred_array, list):
        percent_behavior_bins = []
        for array in pred_array:
            behavior_frames = array.sum()
            percent_behavior = (behavior_frames / len(array))*100
            percent_behavior_bins.append(percent_behavior)
        return percent_behavior_bins
    else:
        raise TypeError(f'pred_array should be a numpy array or a list, found {type(pred_array)} instead')         

def total_behavior(model: Union[str, Path, RandomForestClassifier],
                  features_path: Union[List[str], List[Path]],
                  min_bout_duration_ms: int,
                  fps: int,
                  threshold: float,
                  save_folder: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Calculate percent of behavior across the whole feature file length, for a set of feature files
    Params:
        model (str, Path, RandomForestClassifier): random forest model to use for inference
        features_path (List[str], List[Path]): path to a features file
        min_bout_duration_ms (int): minimum bout duration in milliseconds
        fps (int): Frames per second of the video
        threshold (float): threshold to classify frame as behavior. Default is 0.5
        save_folder (Optional, str, Path): path to a folder to save raw predictions into
    Returns:
        total_behavior_df (pd.DataFrame): DataFrame with behavior percent (of time) across the whole feature file
    """
    total_behavior_df = pd.DataFrame()
    for path in features_path:
        raw_preds = run_inference(model,
                                path,
                                save_folder=save_folder)
        binary_predictions = binary_preds(raw_preds,
                                        threshold=threshold)
        filtered = filter_bouts(binary_predictions,
                                min_bout_duration_ms,
                                fps)
        total_behavior_frames = filtered.sum()
        total_percent_behavior = (total_behavior_frames / len(binary_predictions))*100
        total_duration_behavior = total_behavior_frames / fps

        total_behavior_df.loc[path.stem, f'{config.BEHAVIOR} seconds'] = total_duration_behavior
        total_behavior_df.loc[path.stem, f'{config.BEHAVIOR} percent'] = total_percent_behavior

    return total_behavior_df

def bin_col_key(col: str,
                behavior: Optional[str] = config.BEHAVIOR) -> Tuple:
    """
    Helper function to match and sort columns in bin_behavior_df
    Params:
        col (str): name of the column
        behavior (Optional, str): name of the behavior
    """
    b = re.escape(behavior)
    m = re.fullmatch(rf'{b} bin_(\d+) (seconds|percent)', col)
    if m:
        metric_order = {'seconds':0,
                        'percent':1}
        return (0, int(m.group(1)), int(metric_order[m.group(2)]))
    m = re.fullmatch(rf'{b} total (seconds|percent)', col)
    if m:
        return (1, 0 if m.group(1) == 'seconds' else 1)
    return (2, col)

def bin_behavior(model: Union[str, Path, RandomForestClassifier],
                 features_path: Union[List[str], List[Path]],
                 min_bout_duration_ms: int,
                 fps: int,
                 threshold: float,
                 bin_duration_s: int,
                 save_folder: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Calculate percent of behavior for each user specified time bins on a set of feature files
    Params:
        model (str, Path, RandomForestClassifier): random forest model to use for inference
        features_path (List[str], List[Path]): path to a features file
        min_bout_duration_ms (int): minimum bout duration in milliseconds
        fps (int): Frames per second of the video
        threshold (float): threshold to classify frame as behavior. Default is 0.5
        bin_duration_s (int): Duration of a single time bin in seconds
        save_folder (Optional, str, Path): path to a folder to save raw predictions into
    Returns:
        bin_behavior_df (pd.DataFrame): DataFrame with behavior percent (of time) per bin
    """
    bin_behavior_df = pd.DataFrame()
    for path in features_path:
        raw_preds = run_inference(model,
                                  path,
                                  save_folder=save_folder) # logits for both classes
        binary_predictions = binary_preds(raw_preds,
                                          threshold=threshold) # binary series for positive class (behavior)
        filtered = filter_bouts(binary_predictions,
                                min_bout_duration_ms,
                                fps) # filtered binary series keeping only bouts above min_bout_duration
        bin_predictions = bin_preds(filtered,
                                    bin_duration_s,
                                    fps) # list of binary predictions arrays built from filtered for each bin
        bin_behavior_frames_all = [] # initialize list of behavior length in frames for each bin
        percent_bin_behavior_all = [] # initialize list of behavior percent relative to bin duration for each bin
        for full_pred in bin_predictions: # iterate over each time bin
            total_bin_behavior = full_pred.sum() # total number of frames expressing behavior in that bin
            bin_behavior_frames_all.append(total_bin_behavior)
            percent_bin_behavior = (total_bin_behavior / len(full_pred))*100 # percent of behavior relative to the corresponding bin
            percent_bin_behavior_all.append(percent_bin_behavior)
        bin_behavior_df.loc[path.stem, [f'{config.BEHAVIOR} bin_{n} percent' for n in range(len(percent_bin_behavior_all))]] = percent_bin_behavior_all # assign bin-relative percent behavior
        bin_behavior_df.loc[path.stem, [f'{config.BEHAVIOR} bin_{n} seconds' for n in range(len(bin_behavior_frames_all))]] = [n_frames / fps for n_frames in bin_behavior_frames_all] 
        bin_behavior_frames_all_array = np.array(bin_behavior_frames_all)
        bin_behavior_df.loc[path.stem, f'{config.BEHAVIOR} total percent'] = (bin_behavior_frames_all_array.sum() / len(binary_predictions))*100 # assign total behavior percent
        bin_behavior_df.loc[path.stem, f'{config.BEHAVIOR} total seconds'] = bin_behavior_frames_all_array.sum() / fps
    bin_behavior_df = bin_behavior_df.reindex(columns=sorted(bin_behavior_df.columns, key=bin_col_key))
    bin_behavior_df = bin_behavior_df.apply(pd.to_numeric, errors='coerce')
    return bin_behavior_df

def bout_col_key(col: str,
                 behavior: Optional[str] = config.BEHAVIOR) -> Tuple:
    """
    Helper function to match and sort column names for bout_behavior_df
    Params:
        col (str): name of the column
        behavior (Optional, str): name of the behavior. default is specified in config
    Returns:
        key (tuple): tuple to order columns 
    """
    b = re.escape(behavior)
    m = re.fullmatch(rf'{b} bout_(\d+) duration seconds', col)
    if m:
        return (0, int(m.group(1)))
    if col == rf'{b} n_bouts':
        return (1, 0)
    if col == rf'{b} average_bout_duration':
        return (2, 0)
    if 'seconds' in col and 'total' in col:
        return (3, 0)
    if 'percent' in col and 'total' in col:
        return (4, 0)

    return (5, col)

def bout_behavior(model: Union[str, Path, RandomForestClassifier],
                 features_path: Union[List[str], List[Path]],
                 min_bout_duration_ms: int,
                 fps: int,
                 threshold: float,
                 save_folder: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Calculate bout statistics for behavior across the whole feature file length, for a set of feature files
    Params:
        model (str, Path, RandomForestClassifier): random forest model to use for inference
        features_path (List[str], List[Path]): path to a features file
        min_bout_duration_ms (int): minimum bout duration in milliseconds
        fps (int): frames per second of the video
        threshold (float): threshold to classify a frame as behavior
        save_folder (Optional, str, Path): path to a folder to save raw predictions into
    Returns:
        bout_behavior_df (pd.DataFrame): DataFrame with bout durations and number across the whole feature file
    """
    bout_behavior_df = pd.DataFrame()
    for path in features_path:
        raw_preds = run_inference(model,
                                path,
                                save_folder=save_folder) # logits for both classes
        binary_predictions = binary_preds(raw_preds,
                                        threshold=threshold) # binary series for positive class (behavior)
        filtered = filter_bouts(binary_predictions,
                                min_bout_duration_ms,
                                fps) # filtered binary series keeping only bouts above min_bout_duration
        bout_starts, bout_ends = bout_boundaries(filtered) # start and end frames for each bout, calculated on filtered
        bout_lengths = bout_ends - bout_starts # bout lengths in frames 
        bout_durations = bout_lengths/fps # bout duration in seconds
        n_bouts = len(bout_lengths) # number of bouts
        bout_behavior_df.loc[path.stem, [f'{config.BEHAVIOR} bout_{n} duration seconds' for n in range(len(bout_durations))]] = bout_durations # assign bout durations in seconds
        bout_behavior_df.loc[path.stem, f'{config.BEHAVIOR} n_bouts'] = n_bouts # assign number of bouts
        duration_cols = [col for col in bout_behavior_df.columns if 'duration' in col]
        bout_behavior_df.loc[path.stem, f'{config.BEHAVIOR} total seconds'] = bout_behavior_df.loc[path.stem, duration_cols].sum() # calculate total behavior in seconds by summing all bouts
        bout_behavior_df.loc[path.stem, f'{config.BEHAVIOR} total percent'] = (bout_behavior_df.loc[path.stem, f'{config.BEHAVIOR} total seconds'] / (len(binary_predictions)/fps))*100 # calculate total behavior in percent
    duration_cols = [col for col in bout_behavior_df.columns if 'duration' in col]
    bout_behavior_df[duration_cols] = bout_behavior_df[duration_cols].apply(pd.to_numeric, errors='coerce')
    bout_behavior_df[f'{config.BEHAVIOR} average_bout_duration'] = bout_behavior_df[duration_cols].mean(axis=1, skipna=True) # calculate average bout duration
    bout_behavior_df = bout_behavior_df.reindex(columns=sorted(bout_behavior_df.columns, key=bout_col_key))

    return bout_behavior_df

def bout_bin_col_key(col:str,
                      behavior: Optional[str] = config.BEHAVIOR) -> Tuple:
    """
    Helper function to sort and match bin bouts column names
    Params:
        col (str): column name
        behavior (Optional, str): behavior name
    Returns:
        key (Tuple): tuple to sort columns correctly
    """
    b = re.escape(behavior)
    m = re.fullmatch(rf'{b} bin_(\d+) bout_(\d+) duration seconds', col)
    if m:
        return (0, int(m.group(1)), 0, int(m.group(2)))
    m = re.fullmatch(rf'{b} bin_(\d+) (n_bouts|average_bout_duration seconds|seconds|percent)', col)
    if m:
        metric_order = {'n_bouts': 0,
                        'average_bout_duration seconds': 1,
                        'seconds': 2,
                        'percent': 3
                        }
        return (0, int(m.group(1)), 1, metric_order[m.group(2)])
    m = re.fullmatch(rf'{b} total (seconds|percent)', col)
    if m:
        return (1, 0, 0 if m.group(1) == 'seconds' else 1)

    return (2, 0, 0, col)

def bout_bin_behavior(model: Union[str, Path, RandomForestClassifier],
                      features_path: Union[List[str], List[Path]],
                      min_bout_duration_ms: int,
                      fps: int,
                      threshold: float,
                      bin_duration_s: int,
                      save_folder: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Calculate bout statistics within fixed-duration bins.

    Params:
        model (str, Path, RandomForestClassifier): random forest model to use for inference
        features_path (List[str], List[Path]): paths to feature CSV files
        min_bout_duration_ms (int): minimum bout duration in milliseconds
        fps (int): frames per second of the video
        threshold (float): threshold to classify a frame as behavior
        bin_duration_s (int): duration of a time bin in seconds
        save_folder (Optional, str, Path): folder for raw prediction arrays
    """
    
    rows = []

    for path in features_path:
        raw_preds = run_inference(model, path, save_folder=save_folder)
        binary_predictions = binary_preds(raw_preds, threshold=threshold)
        filtered = filter_bouts(binary_predictions,
                                min_bout_duration_ms,
                                fps)
        bin_predictions = bin_preds(filtered, bin_duration_s, fps) # filtered array split in one array per bin (list of arrays)
        bin_bout_boundaries = bout_boundaries(bin_predictions) # bout starts and ends for each bin (list of tuples)
        bin_bout_lengths = [ends - starts for starts, ends in bin_bout_boundaries] # list of lengths in frames (np.array) of each bout for each bin
        bin_bout_durations = [bin_bout / fps for bin_bout in bin_bout_lengths] # list of duration in seconds (np.array) of each bout in each bin

        row = {"feature": path.stem}
        for i, bin_bout in enumerate(bin_bout_durations):
            row[f"{config.BEHAVIOR} bin_{i} n_bouts"] = int(len(bin_bout))
            row[f"{config.BEHAVIOR} bin_{i} average_bout_duration seconds"] = (
                float(np.mean(bin_bout)) if len(bin_bout) else np.nan
            )
            
            for n, duration in enumerate(bin_bout):
                row[f"{config.BEHAVIOR} bin_{i} bout_{n} duration seconds"] = float(duration)
        for i in range(len(bin_bout_durations)):
            bout_cols_per_bin = [key for key in row if re.fullmatch(rf'{re.escape(config.BEHAVIOR)} bin_{i} bout_\d+ duration seconds', key)]
            row[f'{config.BEHAVIOR} bin_{i} seconds'] = sum([row[col] for col in bout_cols_per_bin])
            row[f'{config.BEHAVIOR} bin_{i} percent'] = (row[f'{config.BEHAVIOR} bin_{i} seconds'] / (len(bin_predictions[i])/fps))*100

        total_bin_cols = [key for key in row if re.fullmatch(rf'{re.escape(config.BEHAVIOR)} bin_\d+ seconds', key)]
        row[f'{config.BEHAVIOR} total seconds'] = sum([row[col] for col in total_bin_cols])
        row[f'{config.BEHAVIOR} total percent'] = (row[f'{config.BEHAVIOR} total seconds'] / (len(filtered)/fps))*100
        rows.append(row)
    

    df = pd.DataFrame(rows).set_index("feature")
    df = df.reindex(columns=sorted(df.columns, key=lambda c: bout_bin_col_key(c, config.BEHAVIOR)))
    return df

def get_px_mm_ratio(video_name: str) -> float:
    """
    Read px to mm ratio for a video
    Params: 
        video_name (str): name of the video file. Can be a tracking of feature file, assuming the names match
    Returns:
        ratio (float): pixel to mm ratio for that video
    """
    full_video_path = config.PATHS['excluded_videos_folder'] / (video_name + '.' + config.VIDEOS['format'])
    video_info = vp.read_video_info(full_video_path)
    height = video_info['height']
    ratio = height / config.VIDEOS['wall_length_mm']
    return ratio

def total_distance(track_folder: Path = config.PATHS['tracks_folder'],
					 ratio: float = None) -> pd.DataFrame:
    track_paths = list(track_folder.glob('*.h5'))
    total_distance_df = pd.DataFrame()
    for path in track_paths:
        track_df = pd.read_hdf(path)
        if config.TRACKS['slice']:
            corrected_df = tp.run_outlier_correction(track_df,
                                                 likelihood_threshold=config.TRACKS['p_cutoff'],
                                                 max_filter=config.TRACKS['max_interpolation'],
                                                 discard_absent=config.TRACKS['discard_animal_absent'],
                                                 reference_bodyparts=config.TRACKS['reference_bodyparts'],
                                                 movement_criterion=config.TRACKS['movement_criterion'],
                                                 location_criterion=config.TRACKS['location_criterion'],
                                                 fps=config.VIDEOS['fps'])[config.TRACKS['slice'][0]:config.TRACKS['slice'][1]]
        else:
            corrected_df = tp.run_outlier_correction(track_df,
                                                 likelihood_threshold=config.TRACKS['p_cutoff'],
                                                 max_filter=config.TRACKS['max_interpolation'],
                                                 discard_absent=config.TRACKS['discard_animal_absent'],
                                                 reference_bodyparts=config.TRACKS['reference_bodyparts'],
                                                 movement_criterion=config.TRACKS['movement_criterion'],
                                                 location_criterion=config.TRACKS['location_criterion'],
                                                 fps=config.VIDEOS['fps'])
            
        corrected_df = corrected_df.copy().sort_index(axis=1)
        centers = corrected_df.loc[:, pd.IndexSlice[:, 'center', ['x','y']]]
        centers.columns.droplevel('scorer')

        x_coords = centers.xs('x', level='coords', axis=1)
        y_coords = centers.xs('y', level='coords', axis=1)

        dx = x_coords.diff()
        dy = y_coords.diff()
        frame_distances = np.sqrt(dx**2 + dy**2) # dataframe with between-frames distances

        name = path.stem
        print('Retrieving px to mm ratio...')
        ratio = get_px_mm_ratio(name)
        print(f'Ratio: {ratio}')
        frame_distances = frame_distances / ratio / 10
        print('Computing total distance')
        distance = frame_distances.sum()
        total_distance_df.loc[path.stem, f'total_distance cm'] = distance.item()

    return total_distance_df

def bin_distance(track_folder: Path = config.PATHS['tracks_folder'],
                 bin_duration_s: int = config.ANALYSIS['bin_duration_s'],
                 fps: int = config.VIDEOS['fps'],
                 ratio: float = None) -> pd.DataFrame:
    track_paths = list(track_folder.glob('*.h5'))
    bin_distance_df = pd.DataFrame()
    for path in track_paths:
        track_df = pd.read_hdf(path)
        if config.TRACKS['slice']:
            corrected_df = tp.run_outlier_correction(track_df,
                                                    likelihood_threshold=config.TRACKS['p_cutoff'],
                                                    max_filter=config.TRACKS['max_interpolation'],
                                                    discard_absent=config.TRACKS['discard_animal_absent'],
                                                    reference_bodyparts=config.TRACKS['reference_bodyparts'],
                                                    movement_criterion=config.TRACKS['movement_criterion'],
                                                    location_criterion=config.TRACKS['location_criterion'],
                                                    fps=config.VIDEOS['fps'])[config.TRACKS['slice'][0]:config.TRACKS['slice'][1]]
        else:
            corrected_df = tp.run_outlier_correction(track_df,
                                                    likelihood_threshold=config.TRACKS['p_cutoff'],
                                                    max_filter=config.TRACKS['max_interpolation'],
                                                    discard_absent=config.TRACKS['discard_animal_absent'],
                                                    reference_bodyparts=config.TRACKS['reference_bodyparts'],
                                                    movement_criterion=config.TRACKS['movement_criterion'],
                                                    location_criterion=config.TRACKS['location_criterion'],
                                                    fps=config.VIDEOS['fps'])
        corrected_df = corrected_df.copy().sort_index(axis=1)
        centers = corrected_df.loc[:, pd.IndexSlice[:, 'center', ['x','y']]]
        centers.columns.droplevel('scorer')

        x_coords = centers.xs('x', level='coords', axis=1)
        y_coords = centers.xs('y', level='coords', axis=1)

        dx = x_coords.diff()
        dy = y_coords.diff()
        frame_distances = np.sqrt(dx**2 + dy**2) # dataframe with between-frames distances

        name = path.stem
        print('Retrieving px to mm ratio...')
        ratio = get_px_mm_ratio(name)
        print(f'Ratio: {ratio}')
        frame_distances = frame_distances / ratio / 10

        frame_distances = frame_distances.values
        frame_window = fps*bin_duration_s
        frame_distances_per_bin = [frame_distances[i:i+frame_window] for i in range(0, len(frame_distances), frame_window)]
        distance_per_bin = [np.nansum(bin_data) for bin_data in frame_distances_per_bin]
        bin_distance_df.loc[path.stem, [f'bin_{i} distance cm' for i in range(len(distance_per_bin))]] = distance_per_bin
    bin_distance_df = bin_distance_df.replace({b'':np.nan, '':np.nan})
    bin_distance_df = bin_distance_df.apply(pd.to_numeric, errors='coerce')
    bin_distance_df[f'total distance cm'] = bin_distance_df.sum(axis=1)

    return bin_distance_df
