from pathlib import Path
import sys
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))
import config
import utils
import os

videos = [str(config.PATHS['videos_folder'] / video) for video in config.VIDEOS['crop_example_videos']]
existing_videos = [os.path.exists(video) for video in videos]
print(existing_videos)
if not all(existing_videos):
    raise IOError(f'The following videos were not found. Please check the crop_example_videos entry in config.py {[video for video, flag in zip(videos, existing_videos) if not flag]}')
utils.write_crop_coords_for_camera(str(config.PROJECT_ROOT / 'crop_coords.csv'),
                                       videos)