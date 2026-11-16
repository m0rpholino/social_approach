# SimBA manager module

# python dependencies
import os, sys, pandas as pd
from pathlib import Path

from simba.utils.config_creator import ProjectConfigCreator
from simba.utils.enums import Paths
from simba.utils.read_write import copy_single_video_to_project
from simba.feature_extractors.feature_extractor_7bp import ExtractFeaturesFrom7bps
from simba.feature_extractors.feature_subsets import FeatureSubsetsCalculator
from simba.outlier_tools.skip_outlier_correction import OutlierCorrectionSkipper
import videoprocessor as vp

# module dependencies
sys.path.append("src")
import tracksprocessor as tp
import config

# defs
def create_project(
	folder: str,
	name: str, 
	clf_name: list,
	bp_count: str,
	bp_config_idx: int,
	animal_count: int,
	filetype: str
	) -> None:
	"""Create the directories of a SimBA project.

	Parameters
	----------
	folder : str
		Folder to create the project into.
	name : str
		Name of the SimBA project.
	clf_name : list
		List of strings for classifier names.
	bp_count : str
		String with the number of bodyparts per animal.
	bp_config_idx : int
		Index of the bodypart config.
	animal_count : int
		Number of animals.
	filetype : str
		Filetype for SimBA outputs.

	Returns
	-------
	None
	"""

	ProjectConfigCreator(
		folder,
		name, 
		clf_name, 
		bp_count, 
		bp_config_idx,
		animal_count,
		filetype
	)

def copy_videos(
	project_config_path: Path,
	excluded_videos_folder: Path
	) -> None:
	"""Copy videos to SimBA project folder.

	Parameters
	----------
	project_config_path : Path
		Path to the SimBA project config.ini.
	excluded_videos_folder : Path
		Path of directory with videos to copy.

	Returns
	-------
	None
	"""

	videos = list(excluded_videos_folder.glob(f"*.{config.VIDEOS['format']}"))
	imported_videos = [video.stem for video in (config.FEATURE_EXTRACTION['project_folder_path'] / 'videos').glob(f"*.{config.VIDEOS['format']}")]
	for video in videos:
		if video.stem in imported_videos:
			continue
		copy_single_video_to_project(
			str(project_config_path),
			str(video),
			symlink=False
		)

def set_video_properties(project_folder: Path,
						 wall_length_mm: float) -> None:
	"""Set video properties in project_folder/logs/video_info.csv. Parameters will be set for all videos in folder.

	Parameters
	----------
	project_folder : Path
		The path to SimBA's project folder.

	Returns
	-------
	None
	"""

	videos_folder = project_folder / 'videos'
	if not any(videos_folder.iterdir()):
		raise IOError('No videos imported in project')

	else:
		video_info_path = project_folder / Paths.VIDEO_INFO.value

		# create dataframe from scratch if file does not exist
		if not Path.exists(video_info_path):
			print('Video info csv was not found. Creating one ...')
			video_info = pd.DataFrame(
				columns=[
					"fps",
					"Resolution_width",
					"Resolution_height",
					"Distance_in_mm",
					"pixels/mm",
				]
			)
			video_info.index.name = 'Video'
		else:
			video_info = pd.read_csv(video_info_path, header=0, index_col=0) # read dataframe

		# loop over videos
		for video in videos_folder.iterdir():
			if video.is_dir():
				continue # iterating over list of Path objects
			video = str(video) # convert to string or get_video_info_ffmpeg breaks

			# get video name
			video_name = Path(video).stem 
			if video_name in video_info.index:
				continue
			else:
				properties = vp.read_video_info(video)

				# set fps
				num, denom = map(int, properties['r_frame_rate'].split('/'))
				fps = num / denom
				video_info.loc[video_name, 'fps'] = fps # must be int for simba consistency

				# set resolution
				video_info.loc[video_name, 'Resolution_width'] = int(properties['width']) # must be int for SimBA consistency
				video_info.loc[video_name, 'Resolution_height'] = int(properties['height']) # must be int for SimBA consistency
				video_info.loc[:, 'Distance_in_mm'] = wall_length_mm
				video_info.loc[:, 'pixels/mm'] = video_info.loc[:, 'Resolution_height'] / video_info.loc[:, 'Distance_in_mm']
				
				# convert column types to int
				video_info['Resolution_width'] = video_info['Resolution_width'].astype('Int64')
				video_info['Resolution_height'] = video_info['Resolution_height'].astype('Int64')
				video_info['fps'] = video_info['fps'].astype('Int64')

		
		video_info.to_csv(video_info_path)

def set_divider_roi(project_folder: Path, test_animal_side: str) -> None:
	"""Set a rectangle ROI over the third of each crop adjacent to the divider.

	When the test animal is on the left, the divider is on the right, and vice
	versa.
	Creates a ROI_definitions.h5 file in the logs folder

	Parameters
	----------
	project_folder : Path
		The path to SimBA's project folder.
	test_animal_side : str
		Side of the cage containing the test animal (``"left"`` or ``"right"``).

	Returns
	-------
	None
	"""

	if test_animal_side not in {'left', 'right'}:
		raise ValueError("test_animal_side must be 'left' or 'right'")

	# read video_info_csv for height and width
	video_info_path = project_folder / Paths.VIDEO_INFO.value
	video_info_df = pd.read_csv(video_info_path, index_col=0)
	px_to_mm = video_info_df['pixels/mm']

	rectangle_width = video_info_df['Resolution_width'] // 3
	if test_animal_side == 'left':
		top_left_x = video_info_df['Resolution_width'] - rectangle_width
		bottom_right_x = video_info_df['Resolution_width']
	else:
		top_left_x = pd.Series(0, index=video_info_df.index)
		bottom_right_x = rectangle_width

	top_left_y = 0
	bottom_right_y = video_info_df['Resolution_height'] # full height for bottom right y
	rectangle_height = video_info_df['Resolution_height']
	rectangle_center_x = top_left_x + (rectangle_width // 2)
	rectangle_center_y = (bottom_right_y - top_left_y) // 2
	rectangle_width_cm = (rectangle_width / px_to_mm) * 100
	rectangle_height_cm = (rectangle_height / px_to_mm) * 100
	area_cm = rectangle_height_cm * rectangle_width_cm
	# build tags dicts
	all_tags = []
	for video in video_info_df.index:
		video_tags = {'Center tag': (rectangle_center_x.loc[video], rectangle_center_y.loc[video]),
						'Top left tag': (top_left_x.loc[video], top_left_y),
						'Bottom right tag' : (bottom_right_x.loc[video], bottom_right_y.loc[video]),
						'Top right tag': (bottom_right_x.loc[video], top_left_y), # top right has same x as bottom right and same y as top left
						'Bottom left tag': (top_left_x.loc[video], bottom_right_y.loc[video]),
						'Top tag': (rectangle_center_x.loc[video], top_left_y), # center point of top side
						'Right tag': (bottom_right_x.loc[video], rectangle_center_y.loc[video]), # center point of right side
						'Left tag': (top_left_x.loc[video], rectangle_center_y.loc[video]), # center point of left side
						'Bottom tag': (rectangle_center_x.loc[video], bottom_right_y.loc[video])} # center point of bottom side
		all_tags.append(video_tags) # store in a list
		
	# initialize df for rectangle
	rectangle_df = pd.DataFrame(index=range(len(video_info_df.index)),
								columns = ['Video',
											'Shape_type',
											'Name',
											'Color name',
											'Color BGR',
											'Thickness',
											'Center_X',
											'Center_Y',
											'topLeftX',
											'topLeftY',
											'Bottom_right_X,'
											'Bottom_right_Y',
											'width',
											'height',
											'width_cm',
											'height_cm',
											'area_cm',
											'Tags',
											'Ear_tag_size'])
	# set values
	rectangle_df['Video'] = list(video_info_df.index) # works?
	rectangle_df['Shape_type'] = 'rectangle' # correct?
	rectangle_df['Name'] = 'divider_roi'
	rectangle_df['Color name'] = 'Red' # hard coding default value, not important
	rectangle_df['Color BGR'] = [(0, 0, 255)] * len(rectangle_df) # hard coding default value, not important
	rectangle_df['Thickness'] = 5 # hard coding default value
	rectangle_df['Center_X'] = rectangle_center_x.values
	rectangle_df['Center_Y'] = rectangle_center_y.values
	rectangle_df['topLeftX'] = top_left_x.values
	rectangle_df['topLeftY'] = top_left_y
	rectangle_df['Bottom_right_X'] = bottom_right_x.values
	rectangle_df['Bottom_right_Y'] = bottom_right_y.values
	rectangle_df['width'] = rectangle_width.values
	rectangle_df['height'] = rectangle_height.values
	rectangle_df['width_cm'] = rectangle_width_cm.values
	rectangle_df['height_cm'] = rectangle_height_cm.values
	rectangle_df['area_cm'] = area_cm.values
	rectangle_df['Tags'] = all_tags
	rectangle_df['Ear_tag_size'] = 15 # hard coding default value

	# initalize empty dicts for other shapes - columns with no index
	circles_df = pd.DataFrame(columns=['Video',
											'Shape_type',
											'Name',
											'Color name',
											'Color BGR',
											'Thickness',
											'centerX',
											'centerY',
											'radius',
											'radius_cm',
											'area_cm',
											'Tags',
											'Ear_tag_size'])
	polygon_df = pd.DataFrame(columns=['Video',
										'Shape_type',
										'Name',
										'Color name',
										'Color BGR',
										'Thickness',
										'Center_X',
										'Center_Y',
										'vertices',
										'center',
										'area',
										'max_vertice_distance',
										'area_cm',
										'Tags',
										'Ear_tag_size'])
	# organize dfs in a dict with keys for saving
	dfs = {r'/circleDf': circles_df,
			r'/polygons': polygon_df,
			r'/rectangles':rectangle_df
			}
	roi_definitions_file_path = project_folder / Path('logs') / Path('measures') / 'ROI_definitions.h5'
	with pd.HDFStore(str(roi_definitions_file_path), mode='w') as store:
		for key, df in dfs.items():
			store.put(key, df, format='fixed', data_columns=True)

def skip_outlier_correction(project_config_path: Path) -> None:
	"""Skip outlier correction using the SimBA skipper.

	Parameters
	----------
	project_folder : Path
		The path to SimBA's project folder.

	Returns
	-------
	None
	"""

	skipper = OutlierCorrectionSkipper(project_config_path)
	skipper.run()

def create_exclusion_project(
	root_folder: Path,
	excluded_videos_folder: Path,
	tracks_folder: Path,
	project_name: str='feature_extraction',
	clf_name: list[str] = ['seeking'],
	bp_count: str = '7',
	bp_config_idx: int = 1,
	animal_count: int = 1,
	filetype: str = 'csv',
	wall_length_mm: float = 600.0,
	test_animal_side: str = 'left'
	) -> None:
	"""Create a SimBA project.

	Parameters
	----------
	root_folder : Path
		Path to the root folder.
	excluded_videos_folder : Path
		Path to the folder with the videos of the excluded rat.
	tracks_folder : Path
		Path to the folder with the tracked bodyparts.
	project_name : str
		Name of the SimBA project
	clf_name : list[str]
		List of strings for classifier names.
	bp_count : str
		Number of bodyparts.
	bp_config_idx : int
		Index of the bodypart config.
	animal_count : int
		Number of animals.
	filetype : str
		Filetype for SimBA outputs.
	wall_length_mm : float
		Length of the exclusion wall.
	test_animal_side : str
		Side of the cage containing the test animal (``"left"`` or ``"right"``).

	Returns
	-------
	None
	"""

	# create project
	print(f'Creating SimBA project "{project_name}" in {root_folder}')
	create_project(
		root_folder,
		project_name,
		clf_name,
		bp_count,
		bp_config_idx,
		animal_count,
		filetype
	)
	project_folder = root_folder / project_name / "project_folder"
	project_config_path = project_folder / "project_config.ini" 

	# import videos
	print(f'Importing videos...')
	copy_videos(
		project_config_path,
		excluded_videos_folder
	)

	# set video properties
	print(f'Setting video_properties...')
	set_video_properties(project_folder)

	# set divider ROI properties
	print(f'Setting divider ROI properties...')
	set_divider_roi(project_folder, test_animal_side)

	# perform outlier correction for each track and save into input_csv folder
	if len(os.listdir(tracks_folder)) == 0:
		raise IOError('No tracking files found in the tracks folder')
	for track in os.listdir(tracks_folder):
		print(f'Performing outlier correction on {track}...')
		track_df = pd.read_hdf(tracks_folder / track)
		corrected_df = tp.run_outlier_correction(track_df)

		# save
		input_csv_path = project_folder / Paths.INPUT_CSV.value
		tp.save(corrected_df, input_csv_path, track[:-3], csv=True)
	
	# skip SimBA outlier correction
	print(f'Skipping simba default outlier correction...')
	skip_outlier_correction(project_config_path)

	print(f'Project created! You can now extract features')

def extract_features(
	project_config_path: Path,
	features_extracted_path: Path,
	subset: list[str] = None,
	) -> None:
	"""Extract features from corrected track files.

	Parameters
	----------
	project_config_path : Path
		The path to SimBA's project config.ini file.
	subset : list[str]
		The features to extract.
	features_extracted_path : Path
		Path to where the extracted features are saved.

	Returns
	-------
	None
	"""
	if not subset:
		ExtractFeaturesFrom7bps(project_config_path).run()
	SubsetCalculator = FeatureSubsetsCalculator(
		project_config_path,
		subset,
		save_dir=features_extracted_path
	)
	SubsetCalculator.run()
