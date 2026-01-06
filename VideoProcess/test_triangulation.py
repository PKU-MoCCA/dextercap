import utils
from post_process import observed_points_per_frame

import numpy as np
import scipy
import json

import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import skimage

import cv2
import time
import os
import textwrap
import glob
from collections import defaultdict
from dataclasses import dataclass
import itertools

import argparse

from typing import List, Tuple


def load_marker_files(filenames:List[str]):
    marker_info = []
    
    for fn in filenames:
        with open(fn) as f:
            info = json.load(f)
            for frame_idx, frame in enumerate(info):
                frame['checked_markers'] = np.asarray(frame['checked_markers'])
                
            marker_info.append(info)
            print(f'loaded marker file {fn}. #frames {len(info)}')
            
    assert len(marker_info) == len(filenames)
    
    return marker_info
    
        
def parse_marker_positions(marker_info, marker_defs:List[Tuple[Tuple[str, int]]]):
    num_cameras = len(marker_info)
    num_frames = min(len(info) for info in marker_info)
    print(f'#cameras: {num_cameras} #frames: {num_frames}')
    
    marker_positions = np.full((num_frames, num_cameras, len(marker_defs), 2), -1)
    
    #        "block_labels": [
    #        {
    #            "label": "4M",
    #            "label_confidence": 0.9977987408638,
    #            "direction": 1,
    #            "dir_confidence": 0.9715937376022339
    #        }, ... ]
    #                   
    #       "blocks": [
    #           [ 1342.0, [ 0, 1, 4, 3]], 
    #          ...]
        
    for cam_idx, info in enumerate(marker_info):
        for frame_idx in range(num_frames):
            frame = info[frame_idx]
            
            markers = frame['checked_markers']
            blocks = frame['blocks']
            block_labels = frame['block_labels']
            
            label_to_block_map = {}
            bad_labels = set()
            # here we validate block labels and construct a list 
            for idx, blk_label in enumerate(block_labels):
                label = blk_label['label']
                if label in ['**', '--']:
                    continue
                
                if label in bad_labels:
                    continue
                
                direction = blk_label['direction']
                if direction == 4:
                    continue
                
                label_confidence = blk_label['label_confidence']                
                dir_confidence = blk_label['dir_confidence']                
                if dir_confidence*label_confidence < 0.8:
                    continue                    
                
                # check if a label appears twice in the same image
                if label in label_to_block_map:
                    # currently, we remove both the blocks... there must be a better way to do this
                    label_to_block_map.pop(label)
                    bad_labels.add(label)
                    continue
                
                    # blk_idx_last = label_to_block_map[label]
                    # last_label_confidence = block_labels[blk_idx_last]['label_confidence']
                    # last_dir_confidence = block_labels[blk_idx_last]['dir_confidence']
                                    
                else:
                    label_to_block_map[label] = idx
            
            # now we add markers based on the predefined patch
            # for each block in the predefined patch, we check if it is in the image, 
            # as that has been stored in `label_to_block_map`
            # note label_to_block_map may contain wrong labels that are not in template, 
            # such label won't be considered in this stage
            for mk_idx, mk_blocks in enumerate(marker_defs):
                mk_pos_candidates = []
                mk_indices_in_candidates_block = []
                mk_indices_in_marker_list = []
                candidate_block_corners = []
                for mk_blk_label, mk_idx_in_block in mk_blocks:
                    blk_idx = label_to_block_map.get(mk_blk_label, -1)
                    if blk_idx < 0:
                        continue
                    
                    blk_area, blk = blocks[blk_idx]
                    blk_label = block_labels[blk_idx]
                    direction = blk_label['direction']
                    
                    # reorder corner index to reflect the direction
                    blk = np.roll(blk, shift=-direction)
                    
                    corner_idx = blk[mk_idx_in_block]
                    corner_pos = markers[corner_idx]
            
                    mk_pos_candidates.append(corner_pos)
                    mk_indices_in_candidates_block.append(mk_idx_in_block)
                    mk_indices_in_marker_list.append(corner_idx)
                    candidate_block_corners.append(markers[blk,:])
                
                # now we have all the candidates, try to compute the actual corner position
                # each block will propose a position for each its corner
                # each marker can be proposed by two blocks, while these two blocks should be
                # adjacent with one and only one shared corner, i.e. the marker.
                # however, when these two blocks are wrongly labeled at separate location, they will 
                # disagree on the positions of the markers. we need to deal with it here.
                if len(mk_pos_candidates) == 0:
                    continue
                
                elif len(mk_pos_candidates) == 1:
                    mk_pos = mk_pos_candidates[0]
                
                elif mk_indices_in_marker_list[0] == mk_indices_in_marker_list[1]:
                    mk_pos = mk_pos_candidates[0]
                
                else: 
                    continue
                    
                    
                marker_positions[frame_idx, cam_idx, mk_idx] = mk_pos
                
    return marker_positions

def animate_marker_positions(marker_positions:np.ndarray, min_valid_value, min_seen_cam,
                             marker_defs, blocks, patches, show_blocks, save_video):
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
            
    #marker_positions (num_frames, num_cameras, len(marker_defs), 2)
    marker_seen_cameras = (marker_positions[...,0] >= min_valid_value).sum(axis=1)
    
    n_col = int(np.ceil(np.sqrt(marker_positions.shape[1])))
    fig, axes = plt.subplots(int(np.ceil(marker_positions.shape[1] / n_col)), n_col, figsize=(15, 15))
    axes = axes.flatten()
    scats = []
    scats_hl = []
    
    hl_markers = marker_seen_cameras[0] >= min_seen_cam
    colors = cm.viridis(np.arange(marker_positions.shape[2]))
    
    if show_blocks:
        patch_names = list(patches.keys())
        patch_color = cm.viridis(np.arange(len(patch_names)) / (len(patch_names)))    
        block_seq = [(blk_def['markers']+[blk_def['markers'][0]], patch_names.index(blk_def['patch'])) for blk_name, blk_def in blocks.items()]    
    block_lines = []
    
    for cam_i in range(marker_positions.shape[1]):        
        all_pos = marker_positions[:,cam_i].reshape(-1, 2)
        xy_min = all_pos[all_pos[:,0]>=min_valid_value].min(axis=0)
        xy_max = all_pos.max(axis=0)
    
        ax = axes[cam_i]
        # scat = ax.scatter(marker_positions[0, cam_i, :, 0], marker_positions[0, cam_i, :, 1], s=1)
        # scats.append(scat)
        scat_hl = ax.scatter(marker_positions[0, cam_i, hl_markers, 0], marker_positions[0, cam_i, hl_markers, 1], s=1, c=colors[hl_markers])
        scats_hl.append(scat_hl)
        
        if show_blocks:
            blk_lines_cam = []
            for (blk_markers, blk_color) in block_seq:
                line_pts = marker_positions[0, cam_i, blk_markers]
                if (line_pts <= min_valid_value).any():
                    line_pts = np.array([[min_valid_value, min_valid_value]])
                line = ax.plot(line_pts[:, 0], line_pts[:, 1], color=patch_color[blk_color], linewidth=1)
                blk_lines_cam.append(line[0])
            block_lines.append(blk_lines_cam)
        
        ax.set(xlim=[xy_min[0], xy_max[0]], ylim=[xy_min[1], xy_max[1]], aspect='equal')
        ax.grid()
    plt.suptitle(f'frame: {0}')   
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=1-0.05, top=1-0.05, wspace=0.05, hspace=0.05)
        
    def update(frame):
        # for each frame, update the data stored on each artist.
        
        hl_markers = marker_seen_cameras[frame] >= min_seen_cam
            
        for cam_i, scat in enumerate(scats_hl):
            x = marker_positions[frame, cam_i, hl_markers, 0]
            y = marker_positions[frame, cam_i, hl_markers, 1]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            scat.set_color(colors[hl_markers])
                        
            if show_blocks:    
                blk_lines_cam = block_lines[cam_i]
                for blk_idx, (blk_markers, blk_color) in enumerate(block_seq):
                    line = blk_lines_cam[blk_idx]
                    
                    line_pts = marker_positions[frame, cam_i, blk_markers]
                    if (line_pts <= min_valid_value).any():
                        line_pts = np.array([[min_valid_value, min_valid_value]])
                    line.set_data(line_pts[:, 0], line_pts[:, 1])
                    line.set_color(patch_color[blk_color])
        
        plt.suptitle(f'frame: {frame}')   
        # return scats+scats_hl+sum(block_lines, [])
    
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=marker_positions.shape[0], interval=30)
    paused = False
    def toggle_pause(event):
        if not event.key in ['d', 'n']:
            return 
        
        nonlocal paused
        if paused:
            ani.resume()
        else:
            ani.pause()
        paused = not paused
    fig.canvas.mpl_connect('button_press_event', toggle_pause)
            
    if save_video is not None and len(save_video) > 0:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=3000)
        ani.save(save_video)
        
    plt.show(block=True)
    
def animate_marker_positions_3d(markers_3d_positions:np.ndarray, min_valid_value, min_seen_cam,
                                marker_defs, blocks, patches, show_blocks, save_video):
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
        
    # markers_3d_positions: (num_frames, num_markers, 3)
    valid_markers = markers_3d_positions[...,0] > min_valid_value
    
    xy_min = markers_3d_positions[valid_markers].min(axis=0)
    xy_max = markers_3d_positions[valid_markers].max(axis=0)
    r = max(xy_max - xy_min)
    c = (xy_max + xy_min) / 2
    xy_max = c + r/3
    xy_min = c - r/3
    
    # print(xy_min, xy_max)
        
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')
        
    ax.axes.set_xlim3d(left=xy_min[0], right=xy_max[0]) 
    ax.axes.set_ylim3d(bottom=xy_min[1], top=xy_max[1]) 
    ax.axes.set_zlim3d(bottom=xy_min[2], top=xy_max[2]) 
    
    markers_to_draw = markers_3d_positions[0, valid_markers[0]]
    scat, = ax.plot(markers_to_draw[:,0], markers_to_draw[:,1], markers_to_draw[:,2], linewidth=1, linestyle='', marker='.', markersize=3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(f'frame: {0}')   
    
    block_lines = []
    if show_blocks:
        patch_names = list(patches.keys())
        patch_color = cm.viridis(np.arange(len(patch_names)) / (len(patch_names)))    
        block_seq = [(blk_def['markers']+[blk_def['markers'][0]], patch_names.index(blk_def['patch'])) for blk_name, blk_def in blocks.items()]  
        for (blk_markers, blk_color) in block_seq:
            line_pts = markers_3d_positions[0, blk_markers]
            if (line_pts <= min_valid_value).any():
                line_pts = np.array([[min_valid_value, min_valid_value, min_valid_value]])
            line = ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], color=patch_color[blk_color])
            block_lines.append(line[0])

    show_frame = 0
    show_frame_inc = 1
    do_step = False
    def update(frame):
        nonlocal show_frame
        nonlocal show_frame_inc
        nonlocal do_step
        show_frame = (show_frame + show_frame_inc) % markers_3d_positions.shape[0]
        
        if do_step:
            show_frame_inc = 0
        
        markers_to_draw = markers_3d_positions[show_frame, valid_markers[show_frame]]
        scat.set_data(markers_to_draw[:,0], markers_to_draw[:,1])
        scat.set_3d_properties(markers_to_draw[:,2])
        
        if show_blocks:
            for idx, (blk_markers, blk_color) in enumerate(block_seq):
                line = block_lines[idx]
                line_pts = markers_3d_positions[show_frame, blk_markers]
                if (line_pts <= min_valid_value).any():
                    line_pts = np.array([[min_valid_value, min_valid_value, min_valid_value]])
                    
                line.set_data(line_pts[:, 0], line_pts[:, 1])
                line.set_3d_properties(line_pts[:, 2])
                line.set_color(patch_color[blk_color])
        
        plt.title(f'frame: {show_frame}')   
        # return [scat]+block_lines

    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=markers_3d_positions.shape[0], interval=30)
    paused = False
    def toggle_pause(event):
        nonlocal do_step
        nonlocal show_frame_inc
        
        if not event.key in ['d', 'n', 'b']:
            return 
        
        if event.key == 'd':
            do_step = False
            show_frame_inc = 0 if show_frame_inc != 0 else 1
            
        else:
            do_step = True
            if event.key == 'n':
                show_frame_inc = 1
            else:
                show_frame_inc = -1
    fig.canvas.mpl_connect('key_press_event', toggle_pause)
    
    
    if save_video is not None and len(save_video) > 0:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=3000)
        ani.save(save_video)
    
    plt.show(block=True)
    

@dataclass
class CameraInfo:
    distort: np.ndarray
    intrinsic: np.ndarray
    rtmat: np.ndarray
    projection: np.ndarray
    
def load_cam_calib_info(cam_calib_folders:List[str]):
    cameras:List[CameraInfo] = []
    for folder in cam_calib_folders:
        if folder == '':
            cameras.append(None)
            continue
        
        distort = np.load(os.path.join(folder, glob.glob('camera_distort*.npy', root_dir=folder)[0]))
        intrinsic = np.load(os.path.join(folder, glob.glob('camera_matrix*.npy', root_dir=folder)[0]))
        rmat = np.load(os.path.join(folder, glob.glob('rmat*.npy', root_dir=folder)[0]))
        tvec = np.load(os.path.join(folder, glob.glob('tvec*.npy', root_dir=folder)[0]))
        rtmat = np.concatenate((rmat, tvec), axis=1)
        projection = np.load(os.path.join(folder, glob.glob('ProjectionMat*.npy', root_dir=folder)[0]))
        
        cameras.append(CameraInfo(distort, intrinsic, rtmat, projection))
        
    return cameras

def load_cam_calib_info_np(cam_param_files:List[str]):
    cameras:List[CameraInfo] = []
    for fn in cam_param_files:        
        params = np.load(fn)
        cam = CameraInfo(distort=params['distorts'],
                   intrinsic=params['intrinsics'],
                   rtmat=np.concatenate((params["rmats"].reshape(3,3), params["tvecs"].reshape(3,1)), axis=-1),
                   projection=params['projection'],
                   )
        cameras.append(cam)
        
    return cameras


def triangulation(marker_positions:np.ndarray, cameras:List[CameraInfo], min_seen_cam:int, 
                  use_ransac:bool, ransac_inliner_thr:int|None=None):
    #marker_positions (num_frames, num_cameras, num_markers, 2)
    assert marker_positions.shape[1] == len(cameras)
    
    #valid_markers (num_frames, num_cameras, num_markers)
    valid_markers = marker_positions[...,0] >= 0
    invalid_markers = marker_positions[...,0] < 0
    
    num_frames, num_cameras, num_markers = marker_positions.shape[:3]
    markers_undistorted = np.empty_like(marker_positions)
    
    for cam_i in range(num_cameras):
        points = cv2.undistortPoints(marker_positions[:,cam_i,:].reshape(-1, 2), 
                                     cameras[cam_i].intrinsic, cameras[cam_i].distort,
                                     P=cameras[cam_i].intrinsic)
        markers_undistorted[:,cam_i] = points.reshape(num_frames, num_markers, 2)
            
    markers_undistorted[invalid_markers, :] = -100
    markers_3d_positions = np.full((num_frames, num_markers, 3), -1000, dtype=marker_positions.dtype)
    
    # re-projection error for each markers
    markers_reproj_error = np.full((num_frames, num_cameras, num_markers), -1.0)
    # cameras that used for computing the position of a markers
    camera_used_flags = np.zeros((num_frames, num_cameras, num_markers), dtype=bool)
        
    # (num_frames, len(marker_defs))
    markers_in_enough_cam = valid_markers.sum(axis=1) >= min_seen_cam
    
    plt.plot(markers_in_enough_cam.sum(-1), 'x-')
    # plt.show()
    t0 = time.time()

    for frame_i in range(num_frames):
        # print(f'solving frame {frame_i}/{num_frames}...')

        pts_in_enough_cam = markers_undistorted[frame_i, :, markers_in_enough_cam[frame_i]]
        masks_in_3_cam = valid_markers[frame_i, :, markers_in_enough_cam[frame_i]]
        pts_3d = np.zeros((pts_in_enough_cam.shape[0], 3))
        
        valid_marker_idx = np.nonzero(markers_in_enough_cam[frame_i])[0]
        
        for m_i in range(pts_in_enough_cam.shape[0]):
            seen_cam = np.nonzero(masks_in_3_cam[m_i])[0]
            P = [cameras[cam_i].projection for cam_i in seen_cam]
            x = pts_in_enough_cam[m_i, seen_cam]
            
            if use_ransac and len(P) > 2:
                pt3d, cam_used, errors = utils.triangulate_robust(P, x, inliner_thr=ransac_inliner_thr)
                pts_3d[m_i] = pt3d
                markers_reproj_error[frame_i, masks_in_3_cam[m_i], valid_marker_idx[m_i]] = errors
                camera_used_flags[frame_i, masks_in_3_cam[m_i], valid_marker_idx[m_i]] = cam_used
            else:
                pts_3d[m_i] = utils.triangulate_nviews(P, x)
                markers_reproj_error[frame_i, masks_in_3_cam[m_i], valid_marker_idx[m_i]] = \
                    utils.triangulate_n_views_error(P, x, pts_3d[m_i])
                camera_used_flags[frame_i, masks_in_3_cam[m_i], valid_marker_idx[m_i]] = True
            
        markers_3d_positions[frame_i, markers_in_enough_cam[frame_i]] = pts_3d

        t1 = time.time()
        if frame_i % 500 == 0:
            print(f'solving frame {frame_i}/{num_frames} took {t1-t0:.2f}s')
        t0 = t1
                
    return markers_undistorted, markers_3d_positions, markers_reproj_error, camera_used_flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-ids', default='1,2,3,4,5,6,7,8,9,10,11,13,14', type=str, help='id of cameras, in the fmt of 2,3,4...')
    parser.add_argument('--label-patches', type=str, required=True, help='label_patches.json')
    parser.add_argument('--marker-files-path', type=str, required=True)
    parser.add_argument('--marker-files-fmt', type=str, default='{0}_voted.json')
    
    parser.add_argument('--cam-param-path', type=str, required=True)
    parser.add_argument('--cam-param-fmt', type=str, default='{0}.npz')
    
    parser.add_argument('-o', '--output', type=str, default='')
    
    parser.add_argument('--min-seen-cam', type=int, default=3, help='if a maker cannot be seen by such number of cameras, it will be neglect')
    parser.add_argument('--marker-reproj-error-thr', type=int, default=10, help='marker error threshold, in px')
    
    parser.add_argument('--use-ransac', default=True, action='store_true')
    parser.add_argument('--ransac-inliner-thr', type=int, default=10, help='inliner threshold, in px')
    
    parser.add_argument('--show-blocks', default=False, action='store_true')
    parser.add_argument('--show-2d-plot', default=False, action='store_true')
    parser.add_argument('--save-2d-video', type=str, default='')
    parser.add_argument('--show-3d-plot', default=False, action='store_true')
    parser.add_argument('--save-3d-video', type=str, default='')
    
    
    argv = ' '.join([
        r'--label-patches dataset/mocap0428/patches_0428.json ',
        r'--marker-files-path dataset/mocap0428/result ',
        r'--cam-param-path camera_params/0514 ',
        r'--ransac-inliner-thr 5',
        r'--marker-reproj-error-thr 5',
        r'-o result/0428/triangulation.npz',
    ]).split()
    
    args = parser.parse_args(argv)
    # args = parser.parse_args()
        
    camera_ids = np.array(list(map(int, args.camera_ids.strip('()"\',').split(','))))
    print(f'camera_ids: {camera_ids}')
    
    # marker definition
    marker_defs, blocks, patches = utils.load_label_patches(args.label_patches)

    # marker positions
    marker_files = [os.path.join(args.marker_files_path, args.marker_files_fmt.format(cam_i)) for cam_i in camera_ids]
    marker_info = load_marker_files(marker_files)
    # print(marker_info)
    marker_positions = parse_marker_positions(marker_info, marker_defs)
    
    # camera parameters
    cam_param_files = [os.path.join(args.cam_param_path, args.cam_param_fmt.format(cam_i)) for cam_i in camera_ids]    
    cameras = load_cam_calib_info_np(cam_param_files)
    
    # now try triangulation
    marker_positions = marker_positions.astype(float)
    markers_undistorted, markers_3d_positions, markers_reproj_error, camera_used_flags = \
        triangulation(marker_positions, cameras, args.min_seen_cam, args.use_ransac, args.ransac_inliner_thr)
        
    # remove markers whose re-projection error is too large in at least one camera
    markers_reproj_error_check = markers_reproj_error[:]
    markers_reproj_error_check[np.logical_not(camera_used_flags)] = -1
    markers_reproj_error_check = markers_reproj_error_check.max(axis=1) # max over all cameras
    invalid_markers = markers_reproj_error_check > args.marker_reproj_error_thr
    markers_3d_positions[invalid_markers] = -1000

    print(markers_3d_positions.shape)

    observed_points_per_frame(markers_3d_positions)
    np.save("result/0428/pts.npy", markers_3d_positions)
    
    if len(args.output) > 0:
        np.savez_compressed(args.output, 
                            marker_positions=marker_positions,          # [nframe, ncamera, npt, 2]
                            markers_undistorted=markers_undistorted,    # [nframe, ncamera, npt, 2]
                            markers_3d_positions=markers_3d_positions,  # [nframe, npt, 3]
                            markers_reproj_error=markers_reproj_error,  # [nframe, ncam, npt]
                            camera_used_flags=camera_used_flags         # [nframe, ncam, npt]
                            )
        
    if args.show_2d_plot:
        animate_marker_positions(markers_undistorted, min_valid_value=-10, min_seen_cam=args.min_seen_cam,
                                marker_defs=marker_defs, blocks=blocks, patches=patches, 
                                show_blocks=args.show_blocks, save_video=args.save_2d_video)
        
    if args.show_3d_plot:
        animate_marker_positions_3d(markers_3d_positions, min_valid_value=-10, min_seen_cam=args.min_seen_cam,
                                    marker_defs=marker_defs, blocks=blocks, patches=patches, 
                                    show_blocks=args.show_blocks, save_video=args.save_3d_video)
    
    
if __name__ == '__main__':
    main()