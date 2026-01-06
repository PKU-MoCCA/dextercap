import utils

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.spatial.transform import Rotation
import torch
from typing import Tuple, List
import roma
import time
import glob
import sys

import argparse


def extract_img_from_video(video_path: str, img_folder: str, interval: int) -> None:
    if os.path.exists(img_folder):
        print(f"Warning: {img_folder} is already existed.")
        return
    else:
        os.makedirs(img_folder, exist_ok=True)
        print(f"create folder: {img_folder}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"can't open：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total frames : {total_frames}")

    save_count = 0
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if current_frame % interval == 0:
            save_path = os.path.join(img_folder, f"{save_count}.png")
            cv2.imwrite(save_path, frame)
            save_count += 1

        current_frame += 1

    cap.release()
    print(f"Saved {save_count} imgs")


def extract_corner(img_path:str, img_filename_fmt:str,
                   frame_range:Tuple[int], interval:int,
                   intrinsics:np.array, distortion:np.array, 
                   chessboard_size:Tuple[int]|List[int], 
                   check_img:bool):
    
    all_corners = []
    # for img_fn in glob.glob('*.png', root_dir=img_path):
    if check_img:        
        os.makedirs(os.path.join(img_path, 'check'), exist_ok=True)

    img_num = frame_range[1] // interval

    for frame_idx in range(img_num):
        img_fn = os.path.join(img_path, img_filename_fmt.format(frame_idx))
        # print(img_fn)
        img = cv2.imread(img_fn)
        
        if distortion is not None:
            # print(img.shape, intrinsics.shape, distortion.shape)
            img = cv2.undistort(img, intrinsics, distortion.reshape(1,5))
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size)
                        
        if not ret:
            corners = np.full((np.prod(chessboard_size),1,2), -1, dtype=np.float32)
        
        all_corners.append(corners)
        if frame_idx % 100 == 0:
            print('processing', img_fn, '...', ret, corners.shape)
        
        if not ret:
            continue

        if check_img:
            img_1 = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            img_1 = cv2.circle(img_1, corners[0,0].astype(int), 10, (255, 250, 12), 3)
            cv2.imwrite(os.path.join(img_path, 'check', f'{frame_idx:04d}.jpg'), img_1)
        
    return all_corners


def extract_all_video_corners(extrinsic_folder:str, 
                              extrinsic_path_fmt:str,
                              img_filename_fmt:str,
                              camera_ids:List[int],
                              frame_range:Tuple[int],
                              interval:int,
                              intrinsics:np.array, 
                              distortions:np.array, 
                              chessboard_size:Tuple[int]|List[int], 
                              check_img:bool):
    
    all_video_corners = []
    for idx, cam_id in enumerate(camera_ids):
        img_path = os.path.join(extrinsic_folder, extrinsic_path_fmt.format(cam_id))
        video_corners = extract_corner(img_path, img_filename_fmt=img_filename_fmt,
                                       frame_range=frame_range,
                                       interval=interval,
                                       intrinsics=intrinsics[idx],
                                       distortion=distortions[idx], 
                                       chessboard_size=chessboard_size,
                                       check_img=check_img
                                       )
        all_video_corners.append(video_corners)

    all_video_corners = np.array(all_video_corners)
    return all_video_corners
    

def extract_ground_coord(img_path:str, img_filename_fmt:str, camera_ids:List[int], 
                         intrinsics:np.array, distortions:np.array, 
                         chessboard_size:Tuple[int]|List[int],
                         chessboard_edge_length:int
                         ):
    
    chessboard_coords = np.zeros((*chessboard_size,3), np.float32).reshape(-1, 3)
    chessboard_coords[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)*chessboard_edge_length
    
    img_points = []
    obj_points = []
    reprojected_points = []
    rvecs = []
    tvecs = []
    
    for idx, cam_id in enumerate(camera_ids):
        img_fn = os.path.join(img_path, img_filename_fmt.format(cam_id))
        if not os.path.exists(img_fn):
            img_points.append(None)
            obj_points.append(None)
            reprojected_points.append(None)
            rvecs.append(np.zeros((3,1)))
            tvecs.append(np.zeros((3,1)))            
            continue
        
        img = cv2.imread(img_fn)            
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size)
        print('processing', img_fn, '...', ret, corners.shape if ret else '')
        
        if not ret:
            img_points.append(None)
            obj_points.append(None)
            reprojected_points.append(None)
            rvecs.append(np.zeros((3,1)))
            tvecs.append(np.zeros((3,1)))
            continue
        
        img_points.append(corners)
        obj_points.append(chessboard_coords)
        
        retval, rvec, tvec = cv2.solvePnP(chessboard_coords, corners, intrinsics[idx], distortions[idx])
        
        reprojcted, _ = cv2.projectPoints(chessboard_coords, rvec, tvec, intrinsics[idx], distortions[idx])
        reprojected_points.append(reprojcted)
        
        rvecs.append(rvec)
        tvecs.append(tvec)
                
    rvecs = np.asarray(rvecs).reshape(len(camera_ids), 3)
    tvecs = np.asarray(tvecs).reshape(len(camera_ids), 3)
    
    
    for idx, cam_id in enumerate(camera_ids):
        print(f'{idx:2d}: camera_{cam_id}')
        if img_points[idx] is None:
            print(f'.... missing')
            continue
        
        reproj_diff = np.linalg.norm(reprojected_points[idx] - img_points[idx], axis=-1)
        print(f'.... rvec: {rvecs[idx]}')
        print(f'.... tvec: {tvecs[idx]}')
        print(f'.... reproj error - max: {reproj_diff.max():0.6f} mean: {reproj_diff.mean()}')
        
    return obj_points, img_points, rvecs, tvecs


def update_unseen_cameras(points_2d:np.ndarray, intrinsics:np.ndarray, rvecs:np.ndarray, tvecs:np.ndarray,
                          camera_ids,
                         chessboard_size:Tuple[int]|List[int],
                         chessboard_edge_length:int):
    
    chessboard_coords = np.zeros((*chessboard_size,3), np.float32).reshape(-1, 3)
    chessboard_coords[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)*chessboard_edge_length
    
    rvecs_zero = (rvecs == 0).all(axis=-1)
    tvecs_zero = (tvecs == 0).all(axis=-1)
    
    unseen_cams = np.nonzero(rvecs_zero)[0]
    seen_cams = np.nonzero(np.logical_not(rvecs_zero))[0]
    
    num_camera, num_frames, num_points = points_2d.shape[:3]
    all_mask = points_2d[:,:,0,0] >= 0
    
    # unseen_cams = [0]
    # seen_cams = [2]
    
    for i in unseen_cams:
        all_hits = []
        for j in seen_cams:
            hit = np.nonzero(np.logical_and(all_mask[i], all_mask[j]))[0]
            if len(hit) > 0:
                all_hits.append(hit.tolist())
            else:
                all_hits.append([])
                
        if len(sum(all_hits, start=[])) == 0:
            print(f'cannot match camera {camera_ids[i]}')
            continue
        
        best_j, best_f = -1, -1
        best_std = 0
        for j, hit in zip(seen_cams, all_hits):
            for f in hit:
                stdi = points_2d[i, f].std()
                stdj = points_2d[j, f].std()
                
                if stdi + stdj > best_std:
                    best_std = stdi + stdj
                    best_j = j
                    best_f = f
                    
        print(f'match camera {camera_ids[i]} to camera {camera_ids[best_j]} @ frame {best_f}')
                            
        retval, rvec_i, tvec_i = cv2.solvePnP(chessboard_coords, points_2d[i, best_f], intrinsics[i], None)
        retval, rvec_j, tvec_j = cv2.solvePnP(chessboard_coords, points_2d[best_j, best_f], intrinsics[best_j], None)
                    
        rmats_i = Rotation.from_rotvec(rvec_i.reshape(3)).as_matrix()
        rmats_j = Rotation.from_rotvec(rvec_j.reshape(3)).as_matrix()
        
        proj_i = np.concatenate((np.concatenate((rmats_i, tvec_i.reshape(3,1)), axis=-1), [[0,0,0,1.]]), axis=0)
        proj_j = np.concatenate((np.concatenate((rmats_j, tvec_j.reshape(3,1)), axis=-1), [[0,0,0,1.]]), axis=0)
        
        rmats_j0 = Rotation.from_rotvec(rvecs[best_j]).as_matrix()
        proj_j0 = np.concatenate((np.concatenate((rmats_j0, tvecs[best_j].reshape(3,1)), axis=-1), [[0,0,0,1.]]), axis=0)
        
        proj_i0 = proj_i @ np.linalg.inv(proj_j) @ proj_j0
        print(proj_i0)
        
        rvec_i0 = Rotation.from_matrix(proj_i0[:3,:3]).as_rotvec()
        tvec_i0 = proj_i0[:3, -1]
        
        rvecs[i] = rvec_i0
        tvecs[i] = tvec_i0
        
    return rvecs, tvecs
        

def compute_3d_point(points_2d:np.ndarray, intrinsics:np.ndarray, rmats:np.ndarray, tvecs:np.ndarray, chessboard_coords:np.ndarray):
    num_camera, num_frames, num_points = points_2d.shape[:3]
    
    points_2d = points_2d.reshape(num_camera, num_frames, num_points, 2)
    all_mask = points_2d[:,:,0,0] >= 0
    all_mask[:, all_mask.sum(axis=0)<2] = False
    
    intrinsics = intrinsics.reshape(num_camera, 3, 3)
    rmats = rmats.reshape(num_camera, 3, 3)
    tvecs = tvecs.reshape(num_camera, 3)
    rtmats = np.concatenate((rmats, tvecs.reshape(num_camera, 3, 1)), axis=-1)
    projections = np.einsum('nij,njk->nik', intrinsics, rtmats)
    
    point_3d = np.zeros((num_frames, num_points, 3), dtype=points_2d.dtype)
    for frame_i in range(num_frames):
        mask = all_mask[:, frame_i]
        valid_camera = np.nonzero(mask)
        P = projections[mask]
        
        for pt_i in range(num_points):
            corners = points_2d[:, frame_i, pt_i, :]
            valid_corners = corners[mask]
                                    
            pt = utils.triangulate_nviews(P, valid_corners)
            point_3d[frame_i, pt_i] = pt
            
    def reproject(pts:np.ndarray):
        pts_homo = pts.reshape(-1, 3)
        pts_homo = np.concatenate((pts_homo, np.ones_like(pts_homo[...,-1:])), axis=-1)
        reprojected:np.ndarray = np.einsum('nij,nlj->nli', projections, pts_homo[None])
        reprojected = reprojected[...,:2] / reprojected[...,-1:]
        reprojected = reprojected.reshape(num_camera, num_frames, num_points, 2)
        
        return reprojected
        
    def rigid_transform(pts_s:np.ndarray, pts_t:np.ndarray):
        # pts_s: (num_point, 3)
        # pts_t: (num_point, 3)
        s_mean = pts_s.mean(axis=0)
        t_mean = pts_t.mean(axis=0)
        rot, _ = Rotation.align_vectors(pts_t-t_mean[None,:], pts_s-s_mean[None,:])
        trans = t_mean - rot.apply(s_mean)
        
        pts_t_1 = rot.apply(pts_s) + trans[None,:]
        
        return rot.as_rotvec().tolist() + trans.tolist(), pts_t_1
    
    results = [rigid_transform(pts_s=chessboard_coords, pts_t=pts) for pts in point_3d]
    chessboard_transforms = np.asarray([r[0] for r in results])
    point_3d_rigid = np.asarray([r[1] for r in results])
    
    reprojected = reproject(point_3d)
    reprojected_rigid = reproject(point_3d_rigid)
            
    return point_3d, reprojected, chessboard_transforms, point_3d_rigid, reprojected_rigid


def bundle_adjust(points_2d:np.ndarray, intrinsics:np.ndarray, rvecs:np.ndarray, tvecs:np.ndarray, 
                         chessboard_size:Tuple[int]|List[int],
                         chessboard_edge_length:int):
    device = 'cuda'
    
    chessboard_coords = np.zeros((*chessboard_size,3), np.float32).reshape(-1, 3)
    chessboard_coords[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)*chessboard_edge_length
    
    
    print(points_2d.shape)
    num_camera, num_frames, num_points = points_2d.shape[:3]
    
    points_2d = points_2d.reshape(num_camera, num_frames, num_points, 2)
    all_mask = points_2d[...,0] >= 0 #
    all_mask = all_mask.reshape(num_camera, -1)
    all_mask[:, all_mask.sum(axis=0)<2] = False
    all_mask = all_mask.reshape(num_camera, num_frames, num_points)
    ##
    
    intrinsics = intrinsics.reshape(num_camera, 3, 3)
    rvecs = rvecs.reshape(num_camera, 3)
    rmats = Rotation.from_rotvec(rvecs).as_matrix().reshape(num_camera, 3, 3)
    tvecs = tvecs.reshape(num_camera, 3)
    
    # compute initial guess of points
    point_3d, _, chessboard_transforms, _, _ = compute_3d_point(points_2d, intrinsics, rmats, tvecs, chessboard_coords)
    
    
    # now do bundle adjust...
    points_2d = torch.from_numpy(points_2d.reshape(num_camera,-1,2).astype(np.float32)).to(device)
    all_mask = torch.from_numpy(all_mask.reshape(num_camera, -1)).to(device)
    intrinsics = torch.from_numpy(intrinsics.astype(np.float32)).to(device)
    
    point_3d = torch.from_numpy(point_3d.reshape(-1,3).astype(np.float32)).to(device)
    chessboard_transforms = torch.from_numpy(chessboard_transforms.reshape(-1,6).astype(np.float32)).to(device)
    chessboard_coords = torch.from_numpy(chessboard_coords.astype(np.float32)).to(device)
    rvecs = torch.from_numpy(rvecs.astype(np.float32)).to(device)
    tvecs = torch.from_numpy(tvecs.astype(np.float32)).to(device)
    
    # # perturb
    # tvecs = torch.einsum('nij,ni->nj', roma.rotvec_to_rotmat(rvecs), tvecs)
    # rvecs += torch.randn_like(rvecs)*0.01
    # tvecs += torch.randn_like(tvecs)*0.1
    # tvecs = torch.einsum('nij,nj->ni', roma.rotvec_to_rotmat(rvecs), tvecs)    
    # # perturb
    
    point_3d_param = torch.nn.Parameter(point_3d.clone(), requires_grad=True).to(device)
    chessboard_transforms_param = torch.nn.Parameter(chessboard_transforms.clone(), requires_grad=True).to(device)
    rvecs_param = torch.nn.Parameter(rvecs.clone(), requires_grad=True).to(device)
    tvecs_param = torch.nn.Parameter(tvecs.clone(), requires_grad=True).to(device)
    intrinsics_param = torch.nn.Parameter(intrinsics.clone(), requires_grad=True).to(device)
    
    optim = torch.optim.AdamW((rvecs_param, tvecs_param), lr=0.001)
    # optim_pts = torch.optim.AdamW((point_3d_param,), lr=0.001)
    optim_pts = torch.optim.AdamW((chessboard_transforms_param,), lr=0.001)
    optim_intrinsics = torch.optim.AdamW((intrinsics_param,), lr=0.001)
    
    optim_sgd = torch.optim.AdamW((rvecs_param, tvecs_param), lr=0.0001)
    optim_pts_sgd = torch.optim.AdamW((point_3d_param,), lr=0.0001)
    optim_intrinsics_sgd = torch.optim.AdamW((intrinsics_param,), lr=0.0001)
    
    t0 = time.time()
    max_epoch = 500
    for iter in range(max_epoch):
        rmat_p = roma.rotvec_to_rotmat(rvecs_param)
        rtmat = torch.cat((rmat_p, tvecs_param.reshape(num_camera, 3, 1)), dim=-1)
        
        if chessboard_transforms_param is None:
            point_3d_update = point_3d_param
        else:
            chessboard_rots = roma.rotvec_to_rotmat(chessboard_transforms_param[:,:3])
            # chessboard_trans = torch.cat((chessboard_rots, chessboard_transforms_param[:,3:].reshape(-1, 3, 1)), dim=-1)
            point_3d_update = torch.einsum('fij,nj->fni', chessboard_rots, chessboard_coords)
            point_3d_update = point_3d_update + chessboard_transforms_param[:,None,3:]            
            point_3d_update = point_3d_update.reshape(point_3d_param.shape)
        
        point_3d_homo = torch.cat((point_3d_update, torch.ones_like(point_3d_update[...,:1])), dim=-1)
        project_mat = intrinsics @ rtmat
        # project_mat = intrinsics_param @ rtmat
        projected_point2d_homo = torch.einsum('nij,nlj->nli', project_mat, point_3d_homo.unsqueeze(0))
        invalid = projected_point2d_homo[...,-1:] == 0
        projected_point2d_homo[...,-1:][invalid] = 1
        
        projected_point2d = projected_point2d_homo / projected_point2d_homo[...,-1:]
        projected_point2d = projected_point2d[...,:2]
        
        loss = torch.nn.functional.mse_loss(projected_point2d[all_mask], points_2d[all_mask])
        
        if iter < max_epoch // 2 or True:
            optim.zero_grad()
            optim_pts.zero_grad()
            optim_intrinsics.zero_grad()

        else:
            optim_sgd.zero_grad()
            optim_pts_sgd.zero_grad()
            optim_intrinsics_sgd.zero_grad()
        
        loss.backward()
        
        if iter < max_epoch // 2 or True:
            optim.step()
            optim_pts.step()
            # optim_intrinsics.step()
        else:
            optim_sgd.step()
            optim_pts_sgd.step()
            # optim_intrinsics_sgd.step()
        
        if time.time() - t0 > 0.5 or True:
            projected_point2d[torch.logical_not(all_mask)] = points_2d[torch.logical_not(all_mask)]
            diff = torch.norm((projected_point2d - points_2d), dim=-1).detach().cpu().numpy()
            max_idx = diff.argmax()
            max_idx = np.unravel_index(max_idx, (num_camera, num_frames, num_points))
            diff_compact = diff[all_mask.detach().cpu().numpy()]
            print(f'epoch: {iter:4d} diff: max {max_idx}, {diff[max_idx[0], max_idx[1]*num_points+max_idx[2]]:0.6f}',
                  f'- std {diff_compact.std():0.6f}  min {diff_compact.min():0.6f} mean {diff_compact.mean():0.6f} median {np.median(diff_compact.flatten()):0.6f} loss: {loss.detach().cpu().numpy()}')
            t0 = time.time()
        
    # new_rmats = roma.rotvec_to_rotmat(rvecs_param).detach().cpu().numpy()
    new_rvecs = rvecs_param.detach().cpu().numpy()
    new_tvecs = tvecs_param.detach().cpu().numpy()
    new_pt3d = point_3d_param.detach().cpu().numpy()
    
    return new_rvecs, new_tvecs, new_pt3d


def check_reproject(reproj_pts, all_mask, start_frame, end_frame,
                   extrinsic_folder:str, 
                    extrinsic_path_fmt:str,
                    img_filename_fmt:str,
                    camera_ids:List[int],
                    intrinsics:np.array, 
                    distortions:np.array, 
                    chessboard_size:Tuple[int]|List[int], 
                   ):
    for idx, cam_id in enumerate(camera_ids):
        img_path = os.path.join(extrinsic_folder, extrinsic_path_fmt.format(cam_id))        
        out_folder = os.path.join(img_path, 'check_reproj')
        os.makedirs(out_folder, exist_ok=True)        
        
        for frame_i in range(start_frame, end_frame):
            corners = (reproj_pts[idx, frame_i-start_frame,:,None,:]).astype(np.float32)
            ret = all_mask[idx, frame_i-start_frame, 0]
            img_fn = os.path.join(img_path, img_filename_fmt.format(frame_i))
            img = cv2.imread(img_fn)
            
            img = cv2.undistort(img, intrinsics[idx], distortions[idx])
        
            if ret:
                img_1 = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            else:
                img_1 = img[:]
                for pt in corners:
                    img_1  = cv2.circle(img_1, pt[0].astype(int), 5, (50, 250, 255), 1)
                
            img_1 = cv2.circle(img_1, corners[0,0].astype(int), 10, (50, 250, 255), 3)
            
            out_img_fn = os.path.join(out_folder, img_filename_fmt.format(frame_i)[:-3]+'jpg')
            cv2.imwrite(out_img_fn, img_1)
            
            print('write', out_img_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-ids', default='1,2,3,4,5,6,7,8,9,10,11,13,14', type=str, help='id of cameras, in the fmt of 2,3,4...')
    parser.add_argument('--chessboard_size', type=str, help='chessboard size in (n,m), note there should be no space', default='(8,11)')
    parser.add_argument('--chessboard_edge_length', type=float, help='length of chessboard edges, in meter', default=0.02)
    parser.add_argument('-o', '--output-path', type=str, required=True)
        
    parser.add_argument('--intrinsic-folder', type=str, required=True)
    # parser.add_argument('--intrinsic-path-fmt', type=str, default='img_{0}', help='a string support .fmt()')
    parser.add_argument('--intrinsic-fn-fmt', type=str, default='{0}.npz', help='a string support .fmt()')
    
    parser.add_argument('--extrinsic-folder', type=str, required=True)
    parser.add_argument('--extrinsic-path-fmt', type=str, default='camera_{0}', help='a string support .fmt()')
    parser.add_argument('--extrinsic-image-fmt', type=str, default='{0}.png', help='a string support .fmt()')
    parser.add_argument('--interval', type=int, default=10, help='interval of frames for learning&testing extrinsics')
    parser.add_argument('--frame-range', type=str, help='range of frames for learning&testing extrinsics, fmt (s,e), or e', required=True)
    parser.add_argument('--train-frame-range', type=str, default='', help='range of frames for learning extrinsics, must within frame-range')
    parser.add_argument('--test-frame-range', type=str, default='', help='range of frames for testing extrinsics, must within frame-range')
    
    parser.add_argument('--use-saved-all-video-corners', type=str, default='', help='a npz file containing all markers extract from all videos, \
                        if not given, the markers will be extracted from the given images')
    
    parser.add_argument('--origin-image-fmt', type=str, default='{0}.png', help='images for determining ground and origin')
    
    parser.add_argument('--save-check-image', action='store_true', default=False)
    
    args = parser.parse_args()

    camera_ids = np.array(list(map(int, args.camera_ids.strip('()"\',').split(','))))
    chessboard_size = tuple(map(int, args.chessboard_size.strip('()"\',').split(',')))
    
    chessboard_coords = np.zeros((*chessboard_size,3), np.float32).reshape(-1, 3)
    chessboard_coords[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)*args.chessboard_edge_length

    for i in camera_ids:
        extract_img_from_video(f"calibration/0514/extrinsic/{i}.avi", f"calibration/0514/extrinsic/camera_{i}", args.interval)
    print("extraction finished")
    
    ## load camera intrinsics
    camera_intrinsics = []
    camera_distortions = []
    for cam_id in camera_ids:
        fn = os.path.join(args.intrinsic_folder, args.intrinsic_fn_fmt.format(cam_id))
        data = np.load(fn)
        camera_intrinsics.append(data['intrinsic_matrix'])
        camera_distortions.append(data['distortion'])
    camera_intrinsics = np.asarray(camera_intrinsics).reshape(len(camera_ids), 3, 3)
    camera_distortions = np.asarray(camera_distortions).reshape(len(camera_ids), -1)
    
    ## load origin images and make an initial guess of the camera extrinsics
    (ground_obj_points, ground_img_points, cam_rvecs_init, cam_tvecs_init) = extract_ground_coord(
        args.extrinsic_folder,  args.origin_image_fmt,  camera_ids,  camera_intrinsics, 
        camera_distortions,  chessboard_size,  args.chessboard_edge_length
    )
        
    ## load all corners
    frame_range = tuple(map(int, args.frame_range.strip('()"\',').split(',')))
    if args.train_frame_range != '':
        train_frame_range = tuple(map(int, args.train_frame_range.strip('()').split(',')))
    else:
        train_frame_range = (0, frame_range[1] - frame_range[0])
        
    if args.test_frame_range != '':
        test_frame_range = tuple(map(int, args.test_frame_range.strip('()').split(',')))
    else:
        test_frame_range = (0, frame_range[1] - frame_range[0])

    print("train_frame_range: ", train_frame_range)
    print("test_frame_range: ", test_frame_range)
    # exit()

    if args.use_saved_all_video_corners == '' or not os.path.exists(args.use_saved_all_video_corners):
        all_video_corners:np.ndarray = extract_all_video_corners (
            args.extrinsic_folder, args.extrinsic_path_fmt, args.extrinsic_image_fmt, camera_ids, frame_range, 
            args.interval, camera_intrinsics, camera_distortions, chessboard_size, check_img=args.save_check_image
        )
        
        save_fn = args.use_saved_all_video_corners 
        if args.use_saved_all_video_corners == '':
            save_fn = os.path.join(args.extrinsic_folder, 'all_video_corners.npz')
                   
        np.savez_compressed(save_fn, all_video_corners=all_video_corners, camera_ids=camera_ids)
    else:
        with np.load(args.use_saved_all_video_corners) as _data:
            all_video_corners:np.ndarray = _data['all_video_corners']
            loaded_camera_ids = _data['camera_ids'] if 'camera_ids' in _data else camera_ids
            assert len(loaded_camera_ids) == len(all_video_corners)
            # loaded_camera_ids = np.array([1,2,3,4,5,6,7,8,9,11,13,14])
            
            cam_set_diff = np.setdiff1d(camera_ids, loaded_camera_ids, assume_unique=True)
            if len(cam_set_diff) > 0:
                add_video_corners = extract_all_video_corners(args.extrinsic_folder,
                                                    args.extrinsic_path_fmt,
                                                    args.extrinsic_image_fmt,
                                                    cam_set_diff,
                                                    frame_range,
                                                    camera_intrinsics,
                                                    camera_distortions,
                                                    chessboard_size,
                                                    check_img=args.save_check_image)
                
                new_array = np.zeros((len(camera_ids), *all_video_corners.shape[1:]), dtype=all_video_corners.dtype)
                
                for idx, cam_id in enumerate(camera_ids):
                    if cam_id in loaded_camera_ids:                        
                        new_array[idx] = all_video_corners[np.nonzero(cam_id == loaded_camera_ids)[0]]
                    else:
                        new_array[idx] = add_video_corners[np.nonzero(cam_id == cam_set_diff)[0]]
                
                all_video_corners = new_array
                
                save_fn = args.use_saved_all_video_corners 
                if args.use_saved_all_video_corners == '':
                    save_fn = os.path.join(args.extrinsic_folder, 'all_video_corners.npz')
                        
                np.savez_compressed(save_fn, all_video_corners=all_video_corners, camera_ids=camera_ids)
                        
    all_video_corners = all_video_corners.squeeze()
    
    
    # ready for training...
    num_points = np.prod(chessboard_size)
    num_camera = len(camera_ids)
    
    train_video_corners = all_video_corners[:, train_frame_range[0]: train_frame_range[1]]
    test_video_corners  = all_video_corners[:,  test_frame_range[0]:  test_frame_range[1]]
    

    train_num_frames = train_video_corners.shape[1]
    
    cam_rvecs_init, cam_tvecs_init = update_unseen_cameras(train_video_corners, intrinsics=camera_intrinsics, 
                                                           rvecs=cam_rvecs_init, tvecs=cam_tvecs_init, 
                                                           camera_ids=camera_ids, chessboard_size=chessboard_size, 
                                                           chessboard_edge_length=args.chessboard_edge_length)
    new_rvects, new_tvecs, new_pt3d = bundle_adjust(train_video_corners, intrinsics=camera_intrinsics, rvecs=cam_rvecs_init, tvecs=cam_tvecs_init,
                                                    chessboard_size=chessboard_size, chessboard_edge_length=args.chessboard_edge_length)
        
    new_cam_trans = -np.einsum('nij,ni->nj', Rotation.from_rotvec(new_rvects).as_matrix(), new_tvecs)
    print(new_tvecs)
    print(new_cam_trans)
    plt.scatter(new_cam_trans[:,0], new_cam_trans[:,1])
    plt.axis('equal')
    # plt.show()
    
    # return
    
    ##################
    # check and test
    new_rmats = Rotation.from_rotvec(new_rvects).as_matrix().reshape(num_camera, 3, 3)
    new_rtmats = np.concatenate((new_rmats,new_tvecs.reshape(num_camera, 3, 1)), axis=-1)
    new_projection = np.einsum('nij,njk->nik', camera_intrinsics, new_rtmats)

    reproj = np.einsum('nij,nlj->nli', new_projection, np.concatenate((new_pt3d, np.ones_like(new_pt3d[...,-1:])), axis=-1)[None])
    reproj = reproj[...,:2]/reproj[...,-1:]
    reproj = reproj.reshape(num_camera,train_num_frames,num_points,2)

    all_mask = train_video_corners[...,0] >= 0
    all_mask = all_mask.reshape(num_camera, -1)
    all_mask[:, all_mask.sum(axis=0)<2] = False
    all_mask = all_mask.reshape(num_camera, -1, num_points)
    reproj_error = (reproj - train_video_corners)
    for err, mask in zip(reproj_error, all_mask):
        diff = np.linalg.norm(err[mask[:,0]], axis=-1)
        print(f'{diff.mean():0.6f}, {diff.std():0.6f}, {diff.max():0.6f}, {diff.min():0.6f}')
    
    plt.plot(reproj_error[all_mask])
    plt.grid()
    plt.savefig('result/train.png')
    # plt.close()

    test_all_mask = test_video_corners[...,0] >= 0
    test_all_mask = test_all_mask.reshape(num_camera, -1)
    test_all_mask[:, test_all_mask.sum(axis=0)<2] = False
    test_all_mask = test_all_mask.reshape(num_camera, -1, num_points)
    test_point_3d, test_reprojected_point2d, test_chessboard_trans, _, test_reprojected_point2d_rigid = compute_3d_point(test_video_corners, intrinsics=camera_intrinsics, rmats=new_rmats, tvecs=new_tvecs, chessboard_coords=chessboard_coords)
    test_errors = test_reprojected_point2d - test_video_corners  # [cam_num, frame_num, point_num, 2]
    print(test_errors.shape, test_all_mask.shape)
    for err, mask in zip(test_errors, test_all_mask):
        diff = np.linalg.norm(err[mask[:,0]], axis=-1)
        print(diff.shape)
        print(f'{diff.mean():0.6f}, {diff.std():0.6f}, {diff.max():0.6f}, {diff.min():0.6f}')
        
    print(test_errors[test_all_mask].shape)
    plt.plot(test_errors[test_all_mask])
    plt.grid()
    plt.savefig('result/test.png')
    # plt.close()
            
    os.makedirs(args.output_path, exist_ok=True)
    for idx, cam_idx in enumerate(camera_ids):
        np.savez_compressed(
            os.path.join(args.output_path, f'{cam_idx}.npz'),
            distorts=camera_distortions[idx],
            intrinsics=camera_intrinsics[idx],
            rmats=new_rmats[idx],
            tvecs=new_tvecs[idx],
            rtmats=new_rtmats[idx],
            projection=new_projection[idx]
        )
        
    if args.save_check_image:
        check_reproject(test_reprojected_point2d_rigid, #test_reprojected_point2d, 
                        test_all_mask, frame_range[0], frame_range[1],
                        args.extrinsic_folder,
                        args.extrinsic_path_fmt,
                        args.extrinsic_image_fmt,
                        camera_ids,
                        camera_intrinsics,
                        camera_distortions,
                        chessboard_size,
                        )


if __name__ == "__main__":
    main()
