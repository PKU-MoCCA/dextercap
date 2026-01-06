# Video Process

This is a deep learning-based pipeline for detecting markers in multi-view videos.

## Directory Structure

```
VideoProcess/
├── models.py                 # Model definitions (UNet, EdgeNet, BlockNet)
├── datasets.py               # Dataset classes
├── train_conv.py             # Train marker detection model (UNet)
├── train_edge.py             # Train edge detection model (EdgeNet)
├── train_block.py            # Train block recognition model (BlockNet)
├── test_conv.py              # Detect markers in videos
├── refine_markers.py         # Subpixel refinement of detected markers
├── test_block.py             # Recognize marker blocks and labels
├── voting.py                 # Voting mechanism for block label correction
├── test_triangulation.py     # Triangulate 3D points from 2D coordinates
├── post_process.py           # Post-processing: outlier removal and interpolation
├── modify_label_file.py      # Modify annotation files path
└── utils.py                  # Utility functions
```

## Main Features

### 1. Model Training

#### Marker Detection Training (`train_conv.py`)

Trains a Cornernet model to detect marker points in images. The model outputs a heatmap indicating marker locations.

**Input:**
- Training data: JSON files containing annotated information
- Dataset configuration: `dataset/all_in_one_data.json`

**Usage:**
```bash
python train_conv.py
```

#### Edge Detection Training (`train_edge.py`)

Trains an EdgeNet model to detect edges between markers, which helps in grouping markers into blocks.

**Input:**
- Training data: JSON files containing annotated information
- Dataset configuration: `dataset/all_in_one_data.json`

**Usage:**
```bash
python train_edge.py
```


#### Block Recognition Training (`train_block.py`)

Trains a BlockNet model to recognize marker blocks and predict their labels (two-character codes) and orientations.

**Input:**
- Training data: JSON files containing annotated information
- Dataset configuration: `dataset/mocapXXXX/label_files.json` (per-session)

**Usage:**
```bash
python train_block.py
```


### 2. Marker and Edge Detection (`test_conv.py`)

Detects marker points and edges in video frames using the trained model.

**Input:**
- Video
- Trained Cornernet and Edgenet checkpoint
- Marker distance threshold (`marker_distance_thr`)

**Usage:**
```bash
python test_conv.py -c <camera_id> --corner-model <cornernet_path> --edge-model <edgenet_path>
```

**Parameters:**
- Modify `<camera_id>` to specify camera index
- Set `marker_distance_thr` to control marker detection sensitivity
- Check threshold statistics in `post_process.py` `statistics()` function

**Output:**
- JSON files containing detected marker positions and edges information for each frame

### 3. Marker Refinement (`refine_markers.py`)

Performs subpixel refinement on detected markers to improve accuracy.

**Usage:**
```bash
python refine_markers.py
```

**Features:**
- Subpixel corner refinement using OpenCV's `cornerSubPix`
- Merges markers that are too close together

**Output:**
- Refined marker positions in JSON format

### 4. Block Recognition (`test_block.py`)

Recognizes marker blocks and predicts their labels using the trained BlockNet model.

**Usage:**
```bash
python test_block.py -c <camera_id> --block-model <blocknet_path> --dataset-def <label_file_json_path>
```

**Parameters:**
- Checkpoint path (`ckpt`)
- Dataset definition path (`dataset-def`)

**Output:**
- JSON files containing detected blocks with predicted labels and orientations

### 5. Voting Mechanism (`voting.py`)

Uses a voting mechanism to check and correct errors in block label recognition by leveraging spatial relationships between blocks.

**Usage:**
```bash
python voting.py
```

**Parameters:**
- Modify date/session name
- Modify camera list

**Algorithm:**
- Uses depth-first search (DFS) to traverse block connections
- Votes for block labels based on patch definitions and spatial relationships
- Corrects mislabeled blocks based on majority voting

**Output:**
- Corrected block labels in JSON format

### 6. Triangulation (`test_triangulation.py`)

Triangulates 3D marker positions from multi-view 2D coordinates using camera parameters.

**Usage:**
```bash
python test_triangulation.py
```

**Input:**
- 2D marker positions from multiple cameras (JSON files)
- Camera intrinsic and extrinsic parameters
- Marker definitions and patch information

**Parameters:**
- Modify arguments in the script (camera IDs, paths, etc.)

**Output:**
- 3D marker positions (`[nframe, npts, 3]`)

### 7. Post-Processing (`post_process.py`)

Post-processes triangulated 3D points to remove outliers and perform interpolation.


**Usage:**
```python
python post_process.py
```

**Usage:**
```python
python post_process.py
```

**Features:**
- Removes outliers using distance-based clustering within patches
- Interpolates missing frames
- Separates hand and object markers based on point indices

**Output:**
- Cleaned 3D marker trajectories
- Interpolated missing data
- Separated hand and object markers

## Workflow


### 1. Model Training

Train the three models:

```bash
python train_conv.py
python train_edge.py
python train_block.py
```


### 2. Marker and Block Detection

Process videos to detect markers and recognize blocks:

```bash
python test_conv.py
python refine_markers.py
python test_block.py
```

### 3. Label Correction

Use voting mechanism to correct block labels:

```bash
python voting.py
```

### 4. 3D Triangulation

Triangulate 3D points from multi-view 2D coordinates:

```bash
python test_triangulation.py
```

### 5. Post-Processing

Clean and interpolate the 3D trajectories:

```bash
python post_process.py
```

## Data Preparation

### Dataset Structure

```
dataset/
├── all_in_one_data.json      # For conv and edge training (all sessions)
└── mocapXXXX/
    ├── origin/                # Original videos/images
    ├── jsons/                 # Annotation JSON files
    ├── imgs/                  # Processed images
    └── label_files.json       # For block training (session-specific)
```


## Notes

1. **Training Data**
   - Ensure annotation files are properly formatted
   - Use `all_in_one_data.json` for conv and edge training (combines all sessions)
   - Use session-specific `label_files.json` for block training

2. **Marker Detection**
   - Adjust `marker_distance_thr` based on marker size and image resolution
   - Check threshold statistics in `post_process.py` `statistics()` function

3. **Triangulation**
   - Requires calibrated camera parameters (from `MocapSystem/`)
   - At least 2 camera views are needed for triangulation. More views improve accuracy and robustness

4. **Post-Processing**
   - Modify part indices according to your patch definitions
   - Adjust outlier removal thresholds based on marker spacing
   - Check for discontinuities and adjust interpolation parameters

