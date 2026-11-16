## track processing module

import pandas as pd, numpy as np
from pathlib import Path

def filter_likelihood(track_df: pd.DataFrame, threshold: float=0.6, max_filter: int=None) -> pd.DataFrame:
	"""Filter pose estimation data based on likelihood. Replaces with NaNs low likelihood estimates.

	Parameters
	----------
	track_df : pd.DataFrame
		Track dataframe.
	threshold : float
		Lower bound of likelihood. Default is 0.6.
	max_filter: int
		Maximum consecutive missing frames to filter out. Default is None (filter out all frames)

	Returns
	-------
	filtered_df : pd.DataFrame
		Filtered DataFrame.
	"""
	# check if track_df is a generic dataframe
	if not isinstance(track_df, pd.DataFrame):
		raise TypeError("argument 'track' should be a DataFrame.")

	filtered_df = track_df.copy().sort_index(axis=1)
	original_df = track_df.copy()

	# get all likelihood columns for all bodyparts
	likelihood = filtered_df.loc[:, pd.IndexSlice[:, :, 'likelihood']]
	mask = likelihood < threshold # this is a booelan multi index dataframe with all likelihood columns, 0 if above threshold 1 unless

	# loop over bodyparts coordinates columns
	for col in filtered_df.loc[:, pd.IndexSlice[:, :, ['x', 'y']]].columns:
		# get likelihood column for each bodypart
		likelihood_col = col[:-1] + ('likelihood',)
		# replace low likelihood values with NaN
		filtered_df.loc[mask[likelihood_col], col] = np.nan # assign nan to all points below likelihood
		if max_filter:
			nan_indices = mask[likelihood_col]# indices of all nan values
			edges = np.diff(nan_indices.astype(int), prepend=0, append=0) # find transition indices between kept and nan values
			segment_starts = np.where(edges == 1)[0] # get all starts of nan segments
			segment_ends = np.where(edges == -1)[0] # get all ends of nan segments
			lengths = segment_ends - segment_starts # get lengths of each segment
			length_mask = lengths > max_filter # flag segments with length above threshold
			starts_to_keep = segment_starts[length_mask] # find starts of segments to keep original data for
			ends_to_keep = segment_ends[length_mask] # find end of segments to keep original data for
			for start, end in zip(starts_to_keep, ends_to_keep): # assign original data back for above-threshold frames
				col_idx = filtered_df.columns.get_loc(col)
				filtered_df.iloc[start:end, col_idx] = original_df[col].iloc[start:end]

	return filtered_df

def get_nan_percentage(track_df: pd.DataFrame) -> pd.DataFrame:
	"""Calculate the percentage of NaN values per body part.

	Parameters
	----------
	track_df : pd.DataFrame
		Track dataframe.

	Returns
	-------
	percentages : pd.DataFrame
		Dataframe with NaN percentage per body part.
	"""
	# use x coordinates to avoid double counting x and y
	x_coords = track_df.xs('x', level='coords', axis=1)

	# count NaNs per video and total frames per video
	nan_counts = x_coords.isna().sum()
	total_counts = x_coords.count() + nan_counts
	
	# calculate percentage
	percentages = (nan_counts / total_counts) * 100
	return percentages

def discard_animal_absent(track_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
	r"""Discard the start of the tracking file until the animal is present. This method should be used after filter_likelihood.
	An animal is considered present if at least half the bodyparts are above pcutoff used in filter_likelihood.

	Parameters
	----------
	track_df : pd.DataFrame
		Track dataframe.
	threshold: float
		Minimum likelihood to count a bodypart as present

	Returns
	-------
	trimmed_df : dict
		Trimmed track dataframe.
	"""

	if not isinstance(track_df, pd.DataFrame):
			raise TypeError('track_df should be a dataframe')
	
	track_df = track_df.sort_index(axis=1)
	bodyparts = track_df.columns.get_level_values('bodyparts').unique()
	tot_bodyparts = len(bodyparts)

	trimmed_track = []

	ind_data = track_df.loc[:, pd.IndexSlice[:, :, ['x', 'y', 'likelihood']]]
	present_count = pd.Series(0, index=track_df.index)
			
	for bp in bodyparts:
		bp_likelihood = ind_data.loc[:, pd.IndexSlice[:, bp, 'likelihood']]
		bp_present = bp_likelihood > threshold
		present_count += bp_present.astype(int).sum(axis=1)

	bp_df = present_count
		
	# Find first frame where all individuals meet criteria for THIS track
	mask = bp_df >= (tot_bodyparts // 2) + 1

	first_valid_frame = mask.loc[mask == True].index[0]
	print("First valid frame: ", first_valid_frame)
			
	# Trim this track from its first valid frame
	trimmed_df = track_df.loc[first_valid_frame:]

	return trimmed_df

def get_outlier_reference(bodyparts: tuple[str, str], method: str, track_df: pd.DataFrame) -> pd.Series:
	r"""Calculate the reference for outlier correction.

	Parameters
	----------
	bodyparts : tuple
		Tuple of bodyparts to calculate distance for.
	method : str
		Use 'mean' or 'median' as reference value.
	track : pd.DataFrame
		Track dataframe.

	Returns
	-------
	reference : pd.Series
		Reference value.
	"""
	if not isinstance(track_df, pd.DataFrame):
			raise TypeError('track_df should be a dataframe')
	
	track_df = track_df.sort_index(axis=1) # sort columns for slicing

	# extract data only for body parts selected
	bp1_x = track_df.loc[:, pd.IndexSlice[:, bodyparts[0], 'x']]
	bp1_y = track_df.loc[:, pd.IndexSlice[:, bodyparts[0], 'y']]
	bp2_x = track_df.loc[:, pd.IndexSlice[:, bodyparts[1], 'x']]
	bp2_y = track_df.loc[:, pd.IndexSlice[:, bodyparts[1], 'y']]

	# calculate distance between bodyparts by frame in pixels
	distance = pd.DataFrame(data=np.sqrt((bp1_x.values - bp2_x.values)**2+(bp1_y.values - bp2_y.values)**2), index=track_df.index)

	# calculate average distance per individual per track
	if method == 'mean':
		reference = distance.mean()
	elif method == 'median':
		reference = distance.median()
	else:
		print(f'Unsupported method: {method}')
	
	return reference

def outlier_correction_movement(reference: pd.Series, criterion: float, track_df: pd.DataFrame) -> pd.DataFrame:
	"""Correct outlier based on movement.

	Parameters
	----------
	reference : pd.Series
		Reference value.
	criterion : float
		Scaling factor for reference value.
	tracks : pd.DataFrame
		Track dataframe.

	Returns
	-------
	corrected_df : pd.DataFrame
		Outlier corrected tracks.
	"""
	if not isinstance(track_df, pd.DataFrame):
		raise TypeError('track_df should be a dataframe')

	track_df = track_df.sort_index(axis=1)
	corrected_df = track_df.copy()

	# calculate threshold
	threshold = reference * criterion

	# select x and y coordinates
	x_coords = track_df.xs('x', level='coords', axis=1)
	y_coords = track_df.xs('y', level='coords', axis=1)

	# calculate movement from one frame to the next
	movement = np.sqrt((x_coords.shift(-1) - x_coords)**2 + (y_coords.shift(-1) - y_coords)**2)

	# define outliers mask
	outliers_mask = movement > threshold[0]
	expanded_mask_coords = pd.concat([outliers_mask, outliers_mask, outliers_mask], 
						keys=['x', 'y', 'likelihood'], 
						axis=1, 
						names=['coords'] + outliers_mask.columns.names)
	level_order = ['scorer', 'bodyparts', 'coords']
	expanded_mask_coords = expanded_mask_coords.reorder_levels(level_order, axis=1)
	expanded_mask_coords.loc[:, pd.IndexSlice[:, :, 'likelihood']] = False
	expanded_mask_coords = expanded_mask_coords.sort_index(axis=1)

	# shift by 1 to match correct outlier frame (the destination one)
	shifted_mask = expanded_mask_coords.shift(1, fill_value=False)

	# replace outlier x and y with nan in corrected copy
	corrected_df.loc[:] = corrected_df.loc[:].where(~shifted_mask, other=np.nan)

	return corrected_df

def outlier_correction_location(reference: pd.Series, criterion: float, track_df: pd.DataFrame) -> pd.DataFrame:
	"""Correct outlier based on location.

	Parameters
	----------
	tracks : pd.DataFrame
		Track dataframe.
	reference : pd.Series
		Series of reference values per track.
	criterion : float
		scaling factor for reference value.

	Returns
	-------
	corrected_df : pd.DataFrame
		Outlier corrected tracks.
	"""
	if not isinstance(track_df, pd.DataFrame):
		raise TypeError('track_df should be a dataframe')
	
	track_df = track_df.sort_index(axis=1)

	# function to apply to each group (track and individual)

	bodyparts = track_df.columns.get_level_values('bodyparts').unique() # get list of bodyparts
	corrected_df = track_df.copy() # create copy to add up masking

	# calculate threshold
	threshold = reference * criterion

	bodypart_masks = {} # initialize bodypart dict to combine masks for each bodypart
	#loop over bodyparts
	for bodypart in bodyparts:
		# get coordinates of bodypart
		bp_x = track_df.loc[:, pd.IndexSlice[:, bodypart, 'x']]
		bp_y = track_df.loc[:, pd.IndexSlice[:, bodypart, 'y']]
		other_bodyparts = [bp for bp in bodyparts if bp != bodypart] # list of all other bodyparts
		distance_df = pd.DataFrame(data=None, index=track_df.index, columns=other_bodyparts) # initialize df of per frame distance to other bodyparts
		for other_bp in other_bodyparts: # loop over bodyparts
			# get coordinate for other bodyparts
			other_bp_x = track_df.loc[:, pd.IndexSlice[:, other_bp, 'x']]
			other_bp_y = track_df.loc[:, pd.IndexSlice[:, other_bp, 'y']]
			# calculate euclidean distance
			distance_df[other_bp] = np.sqrt(
				(other_bp_x.values - bp_x.values)**2 +
				(other_bp_y.values - bp_y.values)**2
			)
		# initialize mask
		mask = distance_df > threshold[0]
		# refine mask to be True when at least two bodyparts are above threshold
		at_least_two = mask.sum(axis=1) >= 2 # hard coding it should be distance > threshold for at lest two bodyparts
		bodypart_masks[bodypart] = at_least_two
	# apply mask and replace with nan
	for bodypart, mask in bodypart_masks.items():
		corrected_df.loc[mask, pd.IndexSlice[:, bodypart, ['x', 'y']]] = np.nan

	return corrected_df

def interpolate(track_df: pd.DataFrame, method: str='linear') -> pd.DataFrame:
	"""Interpolate missing data from track.

	Parameters
	----------
	track_df : pd.DataFrame
		Track dataframe.
	method : str
		Method for interpolation. Default is linear.

	Returns
	-------
	interpolated_df (pd.DataFrame):
		Interpolated track.
	"""
	# check type of tracks
	if not isinstance(track_df, pd.DataFrame):
		raise TypeError('track_df should be a DataFrame')

	track_df = track_df.sort_index(axis=1) # sort index for slicing operations

	coords = track_df.loc[:, pd.IndexSlice[:, :, ['x', 'y']]].copy() # keep only x and y coords, filter out likelihoods
	likelihood = track_df.loc[:, pd.IndexSlice[:, :, ['likelihood']]].copy()

	# interpolate per track
	interpolated_coords = coords.apply(lambda x: x.interpolate(method=method, limit_direction='both'))
	# reassign back the likelihood column
	interpolated_df = pd.concat([interpolated_coords, likelihood], axis=1).sort_index(axis=1)

	return interpolated_df

def smooth(fps: float, time_window: int, method: str, track_df: pd.DataFrame, std: int = 5) -> pd.DataFrame:
	"""Smooths data using rolling window.

	Parameters
	-------
	track_df : pd.DataFrame
		Track dataframe.
	method : str
		Method for interpolation. Default is linear.

	Returns
	-------
	smoothed_df : pd.DataFrame
		Smoothed track.
	"""
	# check type of tracks
	if not isinstance(track_df, pd.DataFrame):
		raise TypeError('track_df should be a DataFrame')

	track_df = track_df.sort_index(axis=1) # sort index for slicing operations
	track_df = track_df.sort_index(axis=0)
	frames_in_window = int(time_window / (1000 / fps)) # get number of frames in window

	coords = track_df.loc[:, pd.IndexSlice[:, :, ['x', 'y']]].copy() # keep only x and y coords, filter out likelihoods
	likelihood = track_df.loc[:, pd.IndexSlice[:, :, ['likelihood']]].copy()

	# smooth per track
	smoothed_coords = coords.apply(lambda x: x.rolling(window=int(frames_in_window), win_type=method, center=True).mean(std=std))
	smoothed_coords[np.isnan(smoothed_coords)] = coords[np.isnan(smoothed_coords)]

	# reassign back the likelihood column
	smoothed_df = pd.concat([smoothed_coords, likelihood], axis=1).sort_index(axis=1)

	return smoothed_df

def reorder_coords(track_df: pd.DataFrame) -> pd.DataFrame:
	"""Reorder coords to x, y, likelihood

	Parameters
	-------
	track_df : pd.DataFrame
		DataFrame to reorder.

	Returns
	-------
	ordered_df : pd.DataFrame
		Track with coordinates reordered.
	"""
	if not isinstance(track_df, pd.DataFrame):
		raise TypeError('track_df should be a dataframe')

	coords_order = ['x', 'y', 'likelihood']
	ordered_df = track_df.reindex(coords_order, level='coords', axis=1)

	return ordered_df

def reorder_columns(track_df: pd.DataFrame, bodypart_order: list[str]) -> pd.DataFrame:
	"""Reorder body parts for specific needs (e.g. SimBA)

	Parameters
	-------
	bodypart_order : List[str]
		Order body parts should appear on.
	track_df : pd.DataFrame
		Dataframe to reorder.

	Returns
	-------
	ordered_df : pd.DataFrame
		Track with body parts reordered.
	"""
	if not isinstance(track_df, pd.DataFrame):
		raise TypeError('track_df should be a dataframe')

	ordered_df = track_df.reindex(bodypart_order, level='bodyparts', axis=1)
	return ordered_df

def run_outlier_correction(
	track_df: pd.DataFrame,
	likelihood_threshold: float,
	max_filter: int,
	discard_absent: bool,
	reference_bodyparts: tuple[str, str],
	movement_criterion: float,
	location_criterion: float,
	fps: float,
	time_window_ms: int = 200,
	bodyparts_order: list = ['left_ear', 'right_ear', 'nose', 'center', 'left_hip', 'right_hip', 'tailbase'],
	) -> pd.DataFrame:
	"""Run full SimBA outlier correction.

	Parameters
	----------
	pcutoff : float
		DeepLabCut likelihood cutoff for removing data
	reference_bodyparts :
		bodyparts to use to calculate reference distance
	movement_criterion :
		scaling factor of reference to detect movement outliers
	location_criterion :
		scaling factor of reference to detect location outliers
	bodyparts_order :
		order of bodyparts for columns
	track_df : pd.DataFrame
		Track dataframe.
	Returns
	-------
	corrected_tracks : pd.DataFrame
		Outlier corrected track.
	"""
	if not isinstance(track_df, pd.DataFrame):
		raise TypeError('track_df should be a dataframe')

	# filter by likelihood
	print(f'\nFiltering out bodyparts below {likelihood_threshold} likelihood...')
	filtered_df = filter_likelihood(track_df, threshold=likelihood_threshold, max_filter=max_filter)
	baseline_nans = get_nan_percentage(filtered_df)
	print(f'Average missing data: {baseline_nans.mean():.3f}')

	# discard frames where animal is absent
	if discard_absent:
		print(f'\nDiscarding frames where animal is absent...')
		trimmed_df = discard_animal_absent(filtered_df, threshold=likelihood_threshold)
	else:
		trimmed_df = filtered_df
	actual_nans = get_nan_percentage(trimmed_df)
	print(f'Missing data after discarding first frames: {actual_nans.mean():.3f}')
	# get reference values
	print(f'\nCalculating reference values for outlier correction...')
	reference = get_outlier_reference(reference_bodyparts, method='mean', track_df=trimmed_df)

	# remove movement outliers
	print(f'\nFinding movement outliers...')
	movement_corrected_df = outlier_correction_movement(reference, movement_criterion, track_df=trimmed_df)

	# remove location outliers
	print(f'\nFinding location outliers...')
	location_corrected_df = outlier_correction_location(reference, location_criterion, track_df=movement_corrected_df)

	# interpolate missing data
	print(f'\nInterpolating outlier values...')
	interpolated_df = interpolate(location_corrected_df)

	# smooth data
	print(f'\nSmoothing data...')
	smoothed_df = smooth(fps, time_window_ms, 'gaussian', track_df=interpolated_df)

	# reoorder coords columns
	print(f'\nReordering columns...')
	reordered_coords_df = reorder_coords(smoothed_df)
	reordered_bodyparts_df = reorder_columns(reordered_coords_df, bodyparts_order)

	print(f'\nOutlier correction complete\n')

	return reordered_bodyparts_df

def save(
	track_df: pd.DataFrame,
	save_folder: Path,
	filename: str,
	csv: bool = None) -> None:
	"""Save a MultiIndex DataFrame into individual h5 files

	Parameters
	-------
	tracks : pd.DataFrame
		Dataframe to save
	save_folder : Path
		Path to the folder to save the files into
	csv : bool
		If True, result will be saved in csv format
	Returns
	-------
		None
	"""
	# save as csv
	if csv:
		full_save_path = Path(save_folder) / Path(filename).with_suffix('.csv')
		track_df.to_csv(full_save_path, index=True, header=True)
	# save as h5
	else:
		full_save_path = Path(save_folder) / Path(filename).with_suffix('.h5')
		track_df.to_hdf(full_save_path, key='df_with_missing', mode='w', format='table')
