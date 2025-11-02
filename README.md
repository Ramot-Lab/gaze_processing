# Gaze Processing & Visualization Toolkit

This project provides a small toolkit to **load, clean, align, analyze, and visualize eye‑tracking (Tobii) data** alongside panel images and synced audio recordings. It also includes helpers to export training data and to create diagnostic plots and videos.

> Main ideas: load per‑participant recordings (gaze `.mat` + audio `.wav` + panel image), clean/interpolate gaze, detect fixations (threshold‑based or model‑based), **correlate fixations with audio events**, compute basic metrics and reliability curves, and render plots/heatmaps/videos.

---

## What's in here?

- `constants.py` — Central config/constants (sampling rates, paths, drawing params, CSV keys). Includes video settings (`VIDEO_CODEC`, `FPS`), screen size (`SCREEN_SIZE`), and conversion (`PIXEL2METER`). Also defines CSV keys like `FIXATION_CSV_KEY_EYE_H/…` and many others used across the code.
- `utils.py` — Small utilities:
  - Manual panel corner picking (`get_panel_edges`), plotting utilities (`plot_points_on_image`),
  - **Fixation detection** (threshold‑based) and helpers (`generate_fixations_threshold_based`, `_calculate_fixation`, `__fixation_finder`),
  - Fixation file reader (`read_fixaiton_data`) that accepts **CSV or NumPy `.npy`** (with fields `t`, `y`, `x`, `evt`).

- `participant_gaze_data_manager.py` — The *core* loader/manager:
  - Builds a per‑participant structure of matched files and metadata.
  - Cleans and interpolates gaze (`interpulate_nan_values`, `clean_outliers_*`).
  - Detects fixations either by **threshold** or via an external **GazeModel** (`generate_fixation_model_based`).
  - **Correlates** fixation and audio timelines (`correlate_fixation_audio_in_time` → returns a unified `DataFrame` with time, eye coords, fixation flag, audio event).

- `visualize_data.py` — Visualization utilities:
  - Heatmaps & gaze overlays on panel images (`show_heatmap`, `plot_gaze_over_img`),
  - Animated videos from gaze (`create_gaze_heatmap_movie`, `show_running_video`),
  - Combine gaze video with the original audio (`add_audio_to_video`, `create_panel_video_with_audio`).

- `process_gaze_data.py` — Small analyses:
  - Compute per‑subject score (`get_score`) and various **declaration/latency** helpers,
  - Distance‑to‑target analyses & plots,
  - Batch analysis/plots per task.

- `reliability_measurement.py` — Reliability functions:
  - `calculate_reliability`, `calculate_reliability_distribution`, and `plot_correlation_scatter` to estimate distribution of correlations over sub‑samples.
  
- `main.py` — Simple entry points:
  - **Data export** to `.npy` for model training (`convert_recordings_to_npy`),
  - Optional trainer runner (see *External code* below).

> Data for the SDMT/KD panels uses a **grid of center locations** for panel keys. The repo includes a `center_locations.json` with the XY centers used by some utilities.


---

## Expected data layout

The manager expects a structure resembling:

<MAIN_DATA_PATH>/
panels_images/
SDMT/ *.jpg # panel images for SDMT
KD/ *.jpg # panel images for KD
<GROUP>/
<PARTICIPANT_ID>/
SDMT/
.mat # Tobii gaze recordings for SDMT
.../.wav # audio recordings (possibly nested)
KD/
.mat
.../.wav
processing_results/ # created by the code if missing