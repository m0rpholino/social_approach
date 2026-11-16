## script to crop excluded side

# initial setup
import sys, os
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))
import config
import videoprocessor as vp

# set path variables

coords_file_path = config.PROJECT_ROOT / 'crop_coords.csv' # check if exists
test_animal_folder = config.PATHS['videos_folder'] / 'test_animal'
test_animal_folder.mkdir(exist_ok=True) # create test animal directory

# get paths to raw videos
videos = list(config.PATHS['videos_folder'].glob(f"*.{config.VIDEOS['format']}"))

if not videos:
	raise IOError(f"No videos of format {config.VIDEOS['format']} found in {config.PATHS['videos_folder']}")
if not coords_file_path.is_file():
	raise IOError(f"No crop_coords file found in {config.PROJECT_ROOT}. You can generate one with get_crop_coordinates.py")
# load crop coordinates
crop_coordinates = vp.load_crop_coordinates(coords_file_path)

# crop videos
cages = [str(n+1) for n in range(config.VIDEOS['cages'])]
video_cameras = [v.stem[0] for v in videos]

for video, camera in zip(videos, video_cameras):
	for cage in cages:
		# if video exists, skip
		if os.path.exists(test_animal_folder / (cage + "__" + video.name)):
			continue

		# load coordinates
		xyxy_coordinates = crop_coordinates[camera + "_" + cage]

		# transform coordinates for cropping the test animal side of cage
		if config.VIDEOS['test_animal_side'] == 'left':
			x = xyxy_coordinates['x1']
		elif config.VIDEOS['test_animal_side'] == 'right':
			x = xyxy_coordinates['x1'] + ((xyxy_coordinates['x2'] - xyxy_coordinates['x1']) // 2)
		else:
			raise ValueError(f"Invalid test animal side: {config.VIDEOS['test_animal_side']}")
		xywh_coordinates = {
			"x": x,
			"y": xyxy_coordinates['y1'],
			"width": (xyxy_coordinates['x2'] - xyxy_coordinates['x1'])//2,
			"height": xyxy_coordinates['y2'] - xyxy_coordinates['y1']
		}
		
		# crop video
		vp.crop(str(video), xywh_coordinates, test_animal_folder / (cage + "__" + video.name))