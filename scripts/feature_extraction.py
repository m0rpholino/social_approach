import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

import simbamanager
import config
import pandas as pd
from simba.utils.enums import Paths
import tracksprocessor as tp

# create project if not existent
if not (config.PROJECT_ROOT / config.FEATURE_EXTRACTION['project_name']).is_dir():
    simbamanager.create_project(config.PROJECT_ROOT,
                                config.FEATURE_EXTRACTION['project_name'],
                                config.FEATURE_EXTRACTION['clf_name'],
                                config.FEATURE_EXTRACTION['bp_count'],
                                config.FEATURE_EXTRACTION['bp_config_idx'],
                                config.FEATURE_EXTRACTION['animal_count'],
                                config.FEATURE_EXTRACTION['filetype'])

simba_project_config_path = config.FEATURE_EXTRACTION['project_config_path']
# copy videos
simbamanager.copy_videos(simba_project_config_path,
                         config.PATHS['excluded_videos_folder']) # will only copy videos that are not already there

# # update video_info_csv
simbamanager.set_video_properties(config.FEATURE_EXTRACTION['project_folder_path'],
                                  config.VIDEOS['wall_length_mm']) # will only add new videos if not already there

# set divider roi
simbamanager.set_divider_roi(config.FEATURE_EXTRACTION['project_folder_path'],
                             config.VIDEOS['test_animal_side']) # recreate it from scratch using video_info_csv which retains all video data

# process and import tracks
tracks = list(config.PATHS['tracks_folder'].glob('*.h5'))
if not tracks:
    raise IOError('No tracking files found in the tracks folder')

input_csv_path = config.FEATURE_EXTRACTION['project_folder_path'] / Paths.INPUT_CSV.value

for track in tracks:
    track_df = pd.read_hdf(track)
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
    tp.save(corrected_df, input_csv_path, track.stem, csv=True)

simbamanager.skip_outlier_correction(simba_project_config_path) # will skip everythin that is in input_csv folder

# extract features
simbamanager.extract_features(config.FEATURE_EXTRACTION['project_config_path'],
                              config.FEATURE_EXTRACTION['features_extracted_path'],
                              config.FEATURE_EXTRACTION['subset'])
