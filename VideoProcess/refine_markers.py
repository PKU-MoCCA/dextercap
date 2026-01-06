import scipy.sparse
import scipy.spatial
import utils

import skimage
import imageio.v3 as iio
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import json
import argparse

import os
import time
import itertools

from typing import List, Tuple
import multiprocessing


def subpixel_refine(gray:np.ndarray, markers:np.ndarray, blocks:np.ndarray, *, win_size:Tuple[int, int]=(5,5), zero_zone:Tuple[int, int]=(-1,-1), refine_iter:int=2):
    if len(markers) == 0:
        return markers, blocks, [], []
    
    if np.issubdtype(gray.dtype, np.integer):
         gray = gray / 255.0
    gray = gray.astype(np.float32)
    
    num_markers = markers.shape[0]
    markers_subpixel = markers.reshape(num_markers, 1, 2).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    
    for iter in range(refine_iter):
        markers_subpixel = cv2.cornerSubPix(gray, markers_subpixel, winSize=win_size, zeroZone=zero_zone, criteria=criteria)
        
        correction = markers_subpixel.squeeze(axis=1) - markers
        # print(f'    subpixel correction {iter+1}/{refine_iter}: max {np.abs(correction).max():0.4f} pixels.')
    
    markers_subpixel = markers_subpixel.squeeze(axis=1).astype(float)
        
    # some markers may move together during this operation, we can use this result to merge mislabeled markers and blocks
    dist = scipy.spatial.distance_matrix(markers_subpixel, markers_subpixel)
    closed_markers = dist < 1.0 # 1 pixel is small enough
    marker_map = np.arange(num_markers)
    marker_to_remove = np.zeros(num_markers, dtype=bool)
    new_num_marker = 0
    for i in range(num_markers):
        if marker_map[i] != i:
            marker_to_remove[i] = True
            continue
        neighbors = np.nonzero(closed_markers[i, i:])[0]
        marker_map[neighbors+i] = new_num_marker
        
        new_num_marker += 1
    
    
    removed_markers = np.nonzero(marker_to_remove)[0]
    removed_blocks = []
    
    assert num_markers - new_num_marker == marker_to_remove.sum()
    if num_markers == num_markers:
        # no marker need to be removed
        blocks_refined = blocks
    else:
        markers_subpixel = markers_subpixel[np.logical_not(marker_to_remove)]
        blocks_refined = []
        block_set = []
        for blk_idx, blk in enumerate(blocks):
            new_blk = [marker_map[i] for i in blk]
            new_blk_ck = tuple(new_blk.sort())
            if new_blk_ck in block_set:
                removed_blocks.append(blk_idx)
                continue
            
            block_set.add(new_blk_ck)
            blocks_refined.append(new_blk)
         
        print(f'    {num_markers - num_markers} markers and {len(blocks) - len(blocks_refined)} blocks are removed.')
        
    
    
    return markers_subpixel, blocks_refined, removed_markers, removed_blocks

def refine(cam_idx):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-video', type=str, required=True)
    parser.add_argument('-m', '--marker-file', type=str, required=True)
    parser.add_argument('-o', '--output-marker-file', type=str, required=True)
    parser.add_argument('-p', '--label-patches', type=str, required=True, help='label_patches.json')
    parser.add_argument('--override', default=False, action='store_true', help='override current marker and block labels')
    parser.add_argument('--save-video', type=str, default='')


    args = ' '.join([
        f'-i dataset/mocap0516/origin/{cam_idx}.avi',
        f'-m dataset/mocap0516/result/{cam_idx}_x.json',
        f'-o dataset/mocap0516/result/{cam_idx}_r.json',
        f'-p dataset/mocap0520/patches_0520.json',
        # f'--save-video dataset/mocap0516/result/{cam_idx}_r.mp4',
    ]).split()
    args = parser.parse_args(args)
    print(f"dataset/mocap0516/result/{cam_idx}_x.mp4")
    # args = parser.parse_args()
    
    with open(args.marker_file) as f:
        all_markers = json.load(f)    
    
    marker_defs, block_defs, patches = utils.load_label_patches(args.label_patches)
    
    out_file = None
    all_marker_pos = np.concatenate([frame['checked_markers'] for frame in all_markers if len(frame['checked_markers']) > 0], axis=0)
    # print(all_marker_pos.shape)
    roi_min = np.maximum(0, np.floor(all_marker_pos.reshape(-1,2).min(axis=0)).astype(int) - 10) // 2 * 2
    roi_max = (np.ceil(all_marker_pos.reshape(-1,2).max(axis=0)).astype(int)) // 2 * 2 + 10
    # print(roi_min, roi_max)
    
    
    # now start to process each frames
    for frame_idx, frame in enumerate(iio.imiter(args.input_video, plugin="pyav")):
        if frame_idx % 1000 == 0: 
            print(f'cam {cam_idx} processing frame {frame_idx}...')
        
        gray = skimage.color.rgb2gray(frame)
        
        markers = np.asarray(all_markers[frame_idx]['checked_markers'])
        blocks = np.asarray([blk[1] for blk in all_markers[frame_idx]['blocks']])
        block_labels = all_markers[frame_idx].get('block_labels', None)
        assert block_labels is None or len(block_labels) == len(blocks)
        
        markers_subpixel, blocks_refined, removed_markers, removed_blocks = subpixel_refine(gray, markers, blocks)
        
        block_labels_refined = block_labels
        if block_labels is not None and len(removed_blocks) > 0:
            block_labels_refined = [lbl for i, lbl in enumerate(block_labels) if not i in removed_blocks]
        
        if args.override:            
            all_markers[frame_idx]['checked_markers'] = markers_subpixel.tolist()
            all_markers[frame_idx]['blocks'] = [(
                cv2.contourArea(markers_subpixel[list(blk)].reshape(-1,1,2).astype(np.float32)), 
                blk.tolist()
                ) for blk in blocks_refined]
            
            if block_labels_refined is not None:
                all_markers[frame_idx]['block_labels'] = block_labels_refined
        else:
            all_markers[frame_idx]['refined_markers'] = markers_subpixel.tolist()
            all_markers[frame_idx]['refined_blocks'] = [(
                cv2.contourArea(markers_subpixel[list(blk)].reshape(-1,1,2).astype(np.float32)), 
                blk.tolist()
                ) for blk in blocks_refined]
            
            if block_labels_refined is not None:
                all_markers[frame_idx]['refined_block_labels'] = block_labels_refined
        
        ## for 
        if len(args.save_video) > 0:
            frame_maker = frame[:]
            block_pts = [np.round(markers_subpixel[list(block_indices)].reshape(-1,1,2)).astype(int) for block_indices in blocks_refined]
            cv2.polylines(frame_maker, block_pts, True, color=[230, 250, 50], thickness=1)
            
            frame_maker = utils.draw_markers(frame_maker, markers, radius=5, color= (180, 180, 55))
            frame_maker = utils.draw_markers(frame_maker, markers_subpixel, radius=5, color=(36, 200, 255))
                    
            frame_maker = frame_maker[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]][:]
            
            if out_file is None:
                out_file = utils.VideoWriter(args.save_video, frame_maker.shape[1], frame_maker.shape[0], fps=20)
            
            out_file.write(frame_maker)

    if out_file is not None:
        out_file.release()
        
    with open(args.output_marker_file, 'w') as f:
        json.dump(all_markers, f)
        
def main():
    cam_list = [1,2,3,4,5,6,7,8]
    with multiprocessing.Pool(processes=len(cam_list)) as pool:
        pool.map(refine, cam_list)


if __name__ == '__main__':
    main()
