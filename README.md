# social approach

`social approach` is a script-based pipeline for quantifying social-approach behaviour and locomotion from cage videos. It combines DeepLabCut pose estimation, custom track cleaning, SimBA feature extraction, and a scikit-learn random-forest classifier.

The workflow is designed for videos containing one or more cages separated into a stimulus-animal side and a test-animal side. It crops the test-animal side of each cage, tracks the animal, extracts frame-level features, trains or applies a behaviour classifier, and writes summary spreadsheets.

In the accompanying forced social exclusion protocol, one test rat is separated from two stimulus rats by a transparent, perforated barrier. Social approach is defined by the nose being less than 2 cm from the barrier, the head directed toward it, and active investigation of the barrier. Grooming and scratching are excluded.

## Pipeline

1. Create the analysis directory structure.
2. Mark cage boundaries on one example video from each camera.
3. Crop the test-animal half of every cage.
4. Train/evaluate a DeepLabCut model or run an existing model on the crops.
5. Clean the resulting pose tracks and extract SimBA features.
6. Train and evaluate a random-forest behaviour classifier.
7. Export social-approach and distance summaries.

The main data flow is:

```text
raw videos
  -> cropped test-animal videos
  -> DeepLabCut .h5 tracks
  -> cleaned tracks and SimBA feature CSVs
  -> random-forest predictions
  -> behaviour and distance .xlsx summaries
```

## Requirements

The repository does not currently include a pinned environment file. The included pipeline requires the following third-party packages and tools:

- DeepLabCut
- SimBA
- OpenCV
- FFmpeg and the `ffmpeg-python` package
- pandas, NumPy, PyTables, and openpyxl
- scikit-learn and joblib
- Matplotlib

DeepLabCut and SimBA have version-specific Python and machine-learning dependencies, so install mutually compatible releases in a dedicated environment. FFmpeg must also be available on `PATH`. A graphical desktop is required for the interactive crop-coordinate tool.

We used Python 3.10.20, DeepLabCut 2.3.8 with the TensorFlow backend, and SimBA 5.4.6. BORIS is used for manual behaviour annotation.

Run every command below from the repository root. The scripts add `src/` to Python's import path themselves; the repository is not packaged for installation.

## Configuration

Edit [`src/config.py`](src/config.py) before running the pipeline.

At minimum, set:

```python
PROJECT_ROOT = Path(r"D:\path\to\analysis")

DLC = {
    "config_path": r"D:\path\to\deeplabcut-project\config.yaml"
}
```

`PROJECT_ROOT` is the data/analysis directory, not necessarily this repository. For the initial `create_project.py` run it must not already exist.

The remaining configuration sections control:

- `BEHAVIOR`: the name used in exported result columns and filenames.
- `VIDEOS`: extension, number of cages per video, frame rate, divider-wall length, and example videos used to define crops.
- `TRACKS`: likelihood filtering, movement/location outlier criteria, body-part references, and an optional frame slice.
- `RF_TRAINING`: model name, grouped cross-validation, parameter grid, scoring, CPU usage, and an optional label slice.
- `RF_INFERENCE`: classification probability threshold and minimum bout length.
- `FEATURE_EXTRACTION`: SimBA project settings and selected feature families.
- `ANALYSIS`: output modes for behaviour and distance, plus the bin duration.

Paths derived from these settings are assembled in `config.py`. If you change `FEATURE_EXTRACTION["project_name"]`, also update `PATHS["features_folder"]` to match `FEATURE_EXTRACTION["features_extracted_path"]`, since the training and evaluation scripts use the former path.

### Input conventions

Several scripts derive metadata from filenames:

- A raw video's first character is treated as its camera identifier.
- `VIDEOS["crop_example_videos"]` must contain exact filenames as they appear under `PROJECT_ROOT/videos` (including the extension).
- Cage crops are named `<cage>__<original-video-name>`.
- DeepLabCut track files are normalized to the cropped-video stem after inference.
- Label CSVs and feature CSVs are paired by sorted filename order during training; use identical stems and keep one label file per feature file.

The default preprocessing and SimBA extractor expect seven body parts named `nose`, `left_ear`, `right_ear`, `center`, `left_hip`, `right_hip`, and `tailbase`. Distance analysis uses `center` and assumes that the cropped video's pixel height corresponds to `VIDEOS["wall_length_mm"]`.

## Project layout

Create a new analysis directory with:

```powershell
python scripts/create_project.py
```

This creates the following structure, with `videos/test_animal/` added when videos are cropped.

```text
PROJECT_ROOT/
|-- videos/
|   `-- test_animal/       # generated cage crops
|-- tracks/                # DeepLabCut HDF5 pose tracks
|-- labels/                # manual frame-event labels
|-- classifiers/           # fitted models and evaluation figures
|-- predictions/           # final Excel summaries
|-- trackers/
`-- logs/
```

Feature extraction later adds `PROJECT_ROOT/feature_extraction/project_folder/`, which contains the SimBA project and its extracted feature CSVs.

## Usage

### 1. Add raw videos

Copy the source videos into `PROJECT_ROOT/videos`. Keep generated crops in the `videos/test_animal` subdirectory; do not place them alongside the raw inputs.

### 2. Record cage coordinates

Set `VIDEOS["crop_example_videos"]` to one representative video per camera, then run:

```powershell
python scripts/get_crop_coordinates.py
```

For each example video, draw one rectangle around every cage:

- drag with the left mouse button to add a rectangle;
- press `x` to remove the most recently drawn rectangle;
- press `c` to accept the coordinates;
- press `q` to quit without completing the file.

The script writes `PROJECT_ROOT/crop_coords.csv`. Cage numbers follow the order in which the rectangles were drawn.

### 3. Crop the test-animal videos

```powershell
python scripts/crop_videos.py
```

For every raw video and configured cage, this script takes the half specified by `VIDEOS["test_animal_side"]` and writes `<cage>__<video>` to `PROJECT_ROOT/videos/test_animal`. Existing crops are skipped.

### 4. Run DeepLabCut

To train or evaluate the project referenced by `DLC["config_path"]`:

```powershell
python scripts/dlc_training.py
python scripts/dlc_evaluation.py
```

To infer pose tracks for crops that do not already have a matching file in `PROJECT_ROOT/tracks`:

```powershell
python scripts/dlc_inference.py
```

The inference script analyzes videos in `videos/test_animal`, moves generated `.h5` files into `tracks`, and removes the DeepLabCut scorer suffix from each track filename. Review DeepLabCut's predictions before continuing; downstream processing assumes valid single-animal tracks with the configured body parts.

### 5. Extract features

For classifier training and evaluation, set `TRACKS["discard_animal_absent"] = False` before extracting features to preserve alignment with the manual labels. Keep `TRACKS["slice"] = ()` when using full-video annotations.

```powershell
python scripts/feature_extraction.py
```

This step:

- creates the configured SimBA project if needed;
- copies the cropped videos into it;
- records video dimensions and physical scale;
- creates a `divider_roi` over the third of each crop adjacent to the divider (rightmost when the test animal is left, and leftmost when it is right);
- filters low-likelihood coordinates, optionally trims the initial animal-absent frames, corrects movement/location outliers, linearly interpolates gaps, and applies 200 ms Gaussian smoothing;
- imports the corrected tracks into SimBA; and
- extracts the feature families listed in `FEATURE_EXTRACTION["subset"]`.

Feature CSVs are written to the configured `features_extracted` directory, normally:

```text
PROJECT_ROOT/feature_extraction/project_folder/csv/features_extracted/
```

### 6. Add behaviour labels

Annotate social approach as a state event in BORIS using the criteria above, without applying a minimum bout duration. Export Tabular Events as one CSV per labelled video into `PROJECT_ROOT/labels`, removing the video extension from each CSV filename. `src/labels.py` expects event rows in chronological start/end pairs and uses these columns.

- `Behavior`: the behaviour name, matching `BEHAVIOR` in `src/config.py`;
- `Image index`: the event frame;
- `Observation duration`: video duration in seconds;
- `FPS`: video frame rate.

Each start/end pair is converted into positive frame labels. Ensure every label file corresponds to exactly one feature CSV and has the same sortable stem.

For full-video labels, leave `RF_TRAINING["label_slice"] = ()` (the default) or use `None`. For equally sized annotated prefixes, use `(0, stop_frame)` and set `TRACKS["slice"]` to the same interval before feature extraction. The current label converter uses the interval length without shifting event indices, so keep the start frame at zero.

### 7. Train the classifier

```powershell
python scripts/train_rf.py
```

The script randomly splits whole videos according to `RF_TRAINING["training_fraction"]` (default 90% training and 10% test) by physically moving matching CSVs into `train/` and `test/` subdirectories. It then performs stratified, group-aware grid search and saves the following files.

```text
classifiers/<model_name>_grid_search.pkl
classifiers/<model_name>_best_estimator.pkl
classifiers/<model_name>_cv_report.png
```

If a previous split exists, the script prompts before replacing it. Because the split is random and no seed is set for file shuffling, exact membership is not reproducible between fresh splits without setting one.

### 8. Evaluate the held-out data

```powershell
python scripts/test_rf.py
```

This loads the best estimator, evaluates the CSVs in the test directories, displays a classification report figure, and saves it as:

```text
classifiers/<model_name>_test_report.png
```

### 9. Export results

For new videos, repeat cropping, DeepLabCut inference, and feature extraction using the trained model's feature settings. The export script reads feature CSVs directly in `features_extracted/`, so CSVs moved into `train/` or `test/` are excluded.

Select the desired modes in `ANALYSIS`, then run:

```powershell
python scripts/behavior_classification.py
```

Supported behaviour modes are:

- `total`: seconds and percent of the analyzed frames;
- `bins`: seconds and percent for each time bin;
- `bouts`: bout count and duration statistics for the analyzed frames;
- `bin_bouts`: bout statistics within each time bin;
- `None`: skip behaviour analysis.

Supported distance modes are `total`, `bins`, and `None`. With the default behavior name, results are written to `PROJECT_ROOT/predictions` as files such as `total_social_approach.xlsx`, `bins_social_approach.xlsx`, `bouts_social_approach.xlsx`, `bins_bouts_social_approach.xlsx`, `total_distance.xlsx`, and `bins_distance.xlsx`.

## Script reference

| Script | Purpose |
| --- | --- |
| `create_project.py` | Create the analysis directory structure. |
| `get_crop_coordinates.py` | Interactively record cage rectangles. |
| `crop_videos.py` | Crop the configured test-animal half of each cage. |
| `dlc_training.py` | Train the configured DeepLabCut network. |
| `dlc_evaluation.py` | Evaluate the configured DeepLabCut network. |
| `dlc_inference.py` | Track cropped videos and collect normalized `.h5` outputs. |
| `feature_extraction.py` | Clean tracks and run SimBA feature extraction. |
| `train_rf.py` | Split labelled videos and tune a random forest. |
| `test_rf.py` | Evaluate the best model on held-out videos. |
| `behavior_classification.py` | Export behaviour and locomotion summaries. |

All scripts listed above are in `scripts/` and are run individually in the workflow order described in Usage.

## Source modules

- `config.py`: project paths and settings for cropping, tracking, feature extraction, classifier training, and analysis.
- `behavior.py`: prediction thresholding, bout filtering, aggregation, and distance calculations.
- `classifiers.py`: metrics, reports, and video-level train/test splitting.
- `labels.py`: conversion of paired label events into frame-level targets.
- `tracksprocessor.py`: likelihood filtering, outlier correction, interpolation, smoothing, and track export.
- `simbamanager.py`: SimBA project creation, video metadata, ROI definitions, and feature extraction.
- `videoprocessor.py`: crop-coordinate loading, FFmpeg crop/trim operations, and video metadata.
- `utils.py`: analysis-directory creation and interactive cage-coordinate selection.

## Important implementation notes

- `train_rf.py` uses `BEHAVIOR` to select positive events from the label files; label values must match that configured name.
- Crop geometry is protocol-specific: the generated test-animal crop is the half selected by `VIDEOS["test_animal_side"]`, while the SimBA divider ROI occupies the third adjacent to the divider (rightmost for a left-side test animal and leftmost for a right-side test animal).
