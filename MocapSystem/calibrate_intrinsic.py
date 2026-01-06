import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
import torch
from typing import Tuple, List
import roma
import time
import glob
import argparse
import sys
   

def extract_frames(video_path: str, interval: int = 20) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame.copy())
        frame_count += 1
    cap.release()
    print(f"Extracted {len(frames)}/{frame_count} frames from {video_path}")
    return frames


def calibrate_camera(frames:List[np.ndarray], chessboard_size:Tuple[int]|List[int], chessboard_edge_length:float):
    
    chessboard_coords = np.zeros((*chessboard_size,3), np.float32).reshape(-1, 3)
    chessboard_coords[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2) * chessboard_edge_length
    
    img_points = []
    obj_points = []

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_ACCURACY)
        
        if not ret:
            continue
        corners = corners[::-1]
        
        img_points.append(corners)
        obj_points.append(chessboard_coords)
        
    
    print(f'calibrating with {len(img_points)}/{len(frames)} frames... ')
    ret, intrinsic_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    
    errors = []
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], intrinsic_matrix, distortion)
        error = np.sqrt((np.asarray(img_points[i] - imgpoints2).squeeze()**2).sum(axis=-1))
        errors.append(error)
    
    errors = np.asarray(errors)
    print( f"total error: mean: {np.mean(errors)} std: {np.std(errors)} max: {np.max(errors)} min: {np.min(errors)} \n")
        
    return ret, intrinsic_matrix, distortion, rvecs, tvecs, errors


def main():
    parser = argparse.ArgumentParser("Intrinsic calibration")
    parser.add_argument('-v', "--video-path", type=str, required=True, help="calibration video path")
    parser.add_argument('--chessboard_size', type=str, help='chessboard size in (n,m), note there should be no space', default='(8,11)')
    parser.add_argument('--chessboard_edge_length', type=float, help='length of chessboard edges, in meter', default=0.02)
    parser.add_argument('--save-check-image', action='store_true', default=False)
    parser.add_argument('-o', '--output-file', type=str, help='a npz file', required=True)
    
    args = parser.parse_args()
        
    chessboard_size = tuple(map(int, args.chessboard_size.strip('()"\',').split(',')))

    print(f'calibrating camera using {args.video_path}...')
    frames = extract_frames(args.video_path, interval=10)

    ret, intrinsic_matrix, distortion, rvecs, tvecs, errors = calibrate_camera(
        frames, chessboard_size=chessboard_size, chessboard_edge_length=args.chessboard_edge_length
    )
    
    np.savez_compressed(args.output_file, intrinsic_matrix=intrinsic_matrix, distortion=distortion, rvecs=rvecs, tvecs=tvecs, errors=errors)


if __name__ == "__main__":
    main()