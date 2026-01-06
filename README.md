# DexterCap: An Affordable and Automated System for Capturing Dexterous Hand-Object Manipulation

Official repository for the paper: [DexterCap: An Affordable and Automated System for Capturing Dexterous Hand-Object Manipulation](https://pku-mocca.github.io/Dextercap-Page/)

## Dataset Usage

### Environment Setup

Install the requirements

```shell
conda create -n HandMocap python=3.10
conda activate HandMocap
pip install -r requirements.txt
```

Register on the [MANO website](https://mano.is.tue.mpg.de/) and download the models.

Then place the models in a folder with the following structure:

```bash
HandReconstruction
└── Data
    └── HumanModels
        └── mano
            ├── MANO_RIGHT.pkl
            └── MANO_LEFT.pkl
```

### Download Dataset

Download through [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), `FILE_NAME` is the name of the dataset file, e.g. `RubiksCube_00-fps_20.npz`:

```
hf download pku-mocca/DexterHand RubiksCube_00-fps_20.npz --repo-type dataset --local-dir ./Dataset/DexterHand
```

Or download the whole dataset from [Hugging Face](https://huggingface.co/datasets/pku-mocca/DexterHand).

```
git clone git@hf.co:datasets/pku-mocca/DexterHand
cd DexterHand
git lfs pull
cd ../
```

### Dataset Visualization

We use [rerun](https://www.rerun.io/) to visualize the result.

For example, to visualize the dataset, run:

```bash
python -m Dataset.visualize --data_path <PATH_TO_NPZ_DATA>
```

To visualize the dataset with hand local coordinates (hand is not moving in global coordinates), run:

```bash
python -m Dataset.visualize --data_path <PATH_TO_NPZ_DATA> --use_hand_local_coordinates True
```

To visualize only a time range (in seconds), use `--start` / `--end`:

```bash
python -m Dataset.visualize --data_path <PATH_TO_NPZ_DATA> --start 5 --end 10
```

## Repo Structure

1. From synchronized multi-view videos, detect or annotate 2D hand/object mocap labels per view.
2. Triangulate multi-view 2D observations to obtain 3D marker trajectories (e.g. `pts_hand`, `pts_obj`).
3. Fit 3D hand markers to the MANO hand model to recover per-frame MANO parameters (`HandReconstruction`).
4. Fit 3D object markers to an object model to recover per-frame pose (or articulated state for Rubik's Cube) (`ObjectReconstruction`).
5. Package hand + object parameters into the released `.npz` dataset format (`Dataset/generate_dataset.py`).

| Stage                 | Input                          | Output                             | Code                          |
| --------------------- | ------------------------------ | ---------------------------------- | ----------------------------- |
| 2D labeling           | multi-view videos              | per-view 2D labels                 | `VideoProcess/`               |
| Triangulation         | 2D labels + camera calibration | 3D markers (`pts_hand`, `pts_obj`) | `MocapSystem/`                |
| Hand Reconstruction   | 3D hand markers                | per-frame MANO params              | `HandReconstruction/`         |
| Object Reconstruction | 3D object markers              | per-frame object pose / state      | `ObjectReconstruction/`       |
| Dataset packaging     | hand + object params           | dataset `.npz`                     | `Dataset/generate_dataset.py` |

For detailed descriptions of each component, please refer to: 
- [`MocapSystem/README.md`](MocapSystem/README.md)
- [`VideoProcess/README.md`](VideoProcess/README.md)
- [`HandReconstruction/README.md`](HandReconstruction/README.md)
- [`ObjectReconstruction/README.md`](ObjectReconstruction/README.md)

## 2D labeling and 3D Triangulation

Please refer to [`VideoProcess/README.md`](VideoProcess/README.md) and [`MocapSystem/README.md`](MocapSystem/README.md).

## Hand and Object Reconstruction

> [!IMPORTANT]
> The public dataset on Hugging Face contains the final reconstructed parameters in `.npz`, but does not include the raw multi-view videos / intermediate 3D marker tracks needed to rerun the full pipeline end-to-end.

### Hand Reconstruction

`HandReconstruction` fits 3D hand markers to the MANO model to recover MANO parameters over time.

Set up the config file, follow the instructions in `HandReconstruction/README.md`.

Then run the following script:

```bash
python -m HandReconstruction.main
```

### Object Reconstruction

`ObjectReconstruction` estimates object pose (rigid objects) or pose + twist state (Rubik's Cube) from 3D object markers.

Follow the instructions in `ObjectReconstruction/README.md`.

### Post Process and Generate Dataset

`Dataset.generate_dataset` merges the hand reconstruction output (MANO params) and the object reconstruction output
(object pose/state) into a single `.npz` file with a consistent schema (the same schema used by `Dataset.visualize`
and the released dataset).

It expects:
- `--hand_param_folder`: a run folder that contains `frame_*.npz` (output of `HandReconstruction`)
- `--object_param_folder`: a run folder that contains exactly one `*.npy` (rigid objects) or `*.npz` (Rubik's Cube)
- per-session metadata in:
  - `HandReconstruction/Result/<mocap_name>/config.py`
  - `ObjectReconstruction/Result/<mocap_name>/config.py`

For example, to generate the dataset of `Cuboid_00`, run:

```bash
python -m Dataset.generate_dataset --mocap_name Cuboid_00 --hand_param_folder <ABSOLUTE_PATH_TO_HAND_DATA_FOLDER> --object_param_folder <ABSOLUTE_PATH_TO_OBJECT_DATA_FOLDER> --save_path <ABSOLUTE_PATH_TO_SAVE_NPZ_DATA>
```

> [!NOTE]
> `mocap_name` is the mocap session name (and folder name), e.g. `Cuboid_00` in `ObjectReconstruction/Result/Cuboid_00/config.py`.

## Citation

If you find our work useful, please cite:

```bibtex
@misc{liang2025dextercap,
  title   = {DexterCap: An Affordable and Automated System for Capturing Dexterous Hand-Object Manipulation},
  author  = {Liang, Yutong and Xu, Shiyi and Zhang, Yulong and Zhan, Bowen and Zhang, He and Liu, Libin},
  journal = {arXiv preprint arXiv:2601.05844},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.05844}
}
```
