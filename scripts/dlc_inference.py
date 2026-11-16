import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))
import config
import deeplabcut as dlc
import shutil
from pathlib import Path

videos = [str(video) for video in ((config.PATHS['videos_folder'] / 'test_animal').glob(f"*.{config.VIDEOS['format']}"))]
tracks = [track.stem for track in config.PATHS['tracks_folder'].glob('*.h5')]
not_analyzed = [video for video in videos if Path(video).stem not in tracks]

dlc.analyze_videos(config.DLC['config_path'],
                   not_analyzed,
                   allow_growth=True)

tracks = list((config.PATHS['videos_folder'] / 'test_animal').glob('*h5'))

for track in tracks:
    shutil.move(track, config.PATHS['tracks_folder'])

moved_tracks = list(config.PATHS['tracks_folder'].glob('*.h5'))

for track in moved_tracks:
    new_name = track.with_stem(track.stem.split('DLC_')[0])
    track.rename(new_name)