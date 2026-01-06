import torch
from torch import nn
from torch.utils.data import DataLoader
import skimage
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from skimage import filters, morphology

import utils
from models import BlockNet
# from models import BlockNet_1 as BlockNet
from datasets import BlockCodeDataset

import cv2

import os
import time
import itertools
import json

import argparse

@torch.no_grad()
def infer_image(model:nn.Module, image:np.array, markers:np.ndarray, blocks:np.ndarray, confidence_thr:float=0.5):
    if len(blocks) == 0:
        return [], [], []
    
    blk_imgs = []
    for blk in blocks:
        corners = markers[blk,:]
        blk_img = dataset.extract_block(image, corners)
        blk_imgs.append(blk_img.astype(np.float32))
        
    blk_imgs = np.stack(blk_imgs, axis=0).reshape(-1, 1, dataset.image_size, dataset.image_size)
    blk_imgs = torch.from_numpy(blk_imgs).to(device)
    
    label_0, label_1, blk_dir = model(blk_imgs)
    label_0 = torch.softmax(label_0, dim=-1)
    label_1 = torch.softmax(label_1, dim=-1)
    blk_dir = torch.softmax(blk_dir, dim=-1)
    
    label_0 = label_0.cpu().numpy()
    label_1 = label_1.cpu().numpy()
    blk_dir = blk_dir.cpu().numpy()
    
    label_0_idx = np.argmax(label_0, axis=-1)
    label_1_idx = np.argmax(label_1, axis=-1)
    blk_dir_idx = np.argmax(blk_dir, axis=-1)
    
    label_0_char = [(dataset.label_characters_0[idx], label_0[i, idx]) for i, idx in enumerate(label_0_idx)]
    label_1_char = [(dataset.label_characters_1[idx], label_1[i, idx]) for i, idx in enumerate(label_1_idx)]
    blk_dir_idx = [(idx, blk_dir[i, idx]) for i, idx in enumerate(blk_dir_idx)]
    
    
    return label_0_char, label_1_char, blk_dir_idx
    
def draw_block_code(image:np.ndarray, markers:np.ndarray, blocks:np.ndarray, label_0_char, label_1_char, blk_dir, wrap_text:bool):
    for blk_idx, blk in enumerate(blocks):
        corners = markers[blk,:]
        label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
        label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]
        
        if label_confidence < 0.5:
            continue
        
        direction = blk_dir[blk_idx][0]
        dir_confidence = blk_dir[blk_idx][1]
        
        pt = np.round(np.mean(corners, axis=0)).astype(int)
        
        if not wrap_text:
            dir_color = dir_confidence * 1.0
            if direction == 0:
                image[max(pt[1]-10, 0):pt[1], pt[0]] = dir_color
            if direction == 2:
                image[pt[1]+1:pt[1]+10, pt[0]] = dir_color
            if direction == 1:
                image[pt[1], max(pt[0]-10, 0):pt[0]] = dir_color
            if direction == 3:
                image[pt[1], pt[0]+1:pt[0]+10] = dir_color
        
        h = 1 if not wrap_text else 0.7
        label_text = label[0] if label[0] in ['-', '*'] else label
        
        if wrap_text:
            text_img_size, _ = cv2.getTextSize('MW', cv2.FONT_HERSHEY_PLAIN, h, 1)
            text_img_size = max(text_img_size) + 1
            text_img = np.zeros((text_img_size,text_img_size), dtype=image.dtype)
            
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, h, 1)
            text_w, text_h = text_size
            
            t_x = max(0, (text_img_size - text_w) // 2)
            t_y = min(max(0, (text_img_size - text_h) // 2 + text_h), text_img_size - 1)
            c = 1.0
            cv2.putText(text_img, label_text, (t_x, t_y), cv2.FONT_HERSHEY_PLAIN, h, c, 1, cv2.LINE_AA)
                            
            # Set up the destination points for the perspective transform
            src = np.array([
                [0, 0],
                [text_img_size - 1, 0],
                [text_img_size - 1, text_img_size - 1],
                [0, text_img_size - 1]], dtype=np.float32)
            
            src = np.roll(src, shift=direction, axis=0)                
                            
            xy_min = np.floor(corners.min(axis=0)).astype(int)
            xy_max = np.ceil(corners.max(axis=0)).astype(int)
            blk_size = max(xy_max - xy_min)

            # Calculate the perspective transform matrix and apply it
            M = cv2.getPerspectiveTransform(src, (corners - xy_min).astype(np.float32).reshape(-1,2))
            # warped = cv2.warpPerspective(text_img, M, (blk_size, blk_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped = cv2.warpPerspective(text_img, M, (blk_size, blk_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped = np.clip(warped, 0, 1)
                        
            img_blk = image[xy_min[1]:xy_min[1]+warped.shape[0],
                            xy_min[0]:xy_min[0]+warped.shape[1]]
            
            text_mask = warped > 0            
            img_blk[text_mask] = img_blk[text_mask]*(1-warped[text_mask]) + warped[text_mask]
            
        else:
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, h, 1)
            text_w, text_h = text_size
            
            c = label_confidence * 1.0
            text_img = np.zeros((text_h+1, text_w+1), dtype=image.dtype)
            cv2.putText(text_img, label_text, (0, text_h), cv2.FONT_HERSHEY_PLAIN, h, c, 1, cv2.LINE_AA)
            
            img_blk = image[max(0, pt[1]-text_img.shape[0]//2):pt[1]+text_img.shape[0]//2,
                            max(0, pt[0]-text_img.shape[1]//2):pt[0]+text_img.shape[1]//2]
            text_img = text_img[:img_blk.shape[0], :img_blk.shape[1]]
            text_mask = text_img > 0
            
            img_blk[text_mask] = text_img[text_mask]
            img_blk[:] = text_img
        
    return image


def draw_block_corners(image:np.ndarray, markers:np.ndarray, blocks:np.ndarray, label_0_char, label_1_char, blk_dir):
    h = 0.5
    for blk_idx, blk in enumerate(blocks):
        corners = markers[blk,:]
        label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
        label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]
        
        direction = blk_dir[blk_idx][0]
        dir_confidence = blk_dir[blk_idx][1]
        
        code_idx = dataset.code_index(label)
        if code_idx < 0: # wrong code, '*', or '-'
            continue
        
        if direction == 4:
            continue
        
        corners = np.roll(corners, shift=-direction, axis=0)
        center = np.mean(corners, axis=0)
        blk_img = dataset.extract_block(image, corners)
        # print(code_idx, label, direction)
        # plt.imshow(blk_img)
        # plt.show()
        
        utils.draw_markers(image, corners, radius=1, color=1, inplace=True)
        
        c = 1
        for i, pt in enumerate(corners):
            pt_id = f'{code_idx}:{i}'
            text_size, _ = cv2.getTextSize(pt_id, cv2.FONT_HERSHEY_PLAIN, h, 1)
            text_w, text_h = text_size
            pt = pt.round().astype(int)
            rel = pt - center
            
            x,y = pt
            
            if rel[0] < 0:
                x = x + 1
            else:
                x = x - text_w
                
            if rel[1] > 0:
                y = y - text_h
            else:
                y = y + text_h
            
            cv2.putText(image, pt_id, (x, y), cv2.FONT_HERSHEY_PLAIN, h, c, 1, cv2.LINE_AA)
            
    return image
        
        
def test_video(model:nn.Module, fn, out_fn, marker_fn, marker_out_fn:str|None=None):
    import cv2
    import ffmpeg
    out_file = None
    
    with open(marker_fn) as f:
        all_markers = json.load(f)
            
    all_marker_pos = np.concatenate([frame['checked_markers'] for frame in all_markers if len(frame['checked_markers']) > 0], axis=0)
    # print(all_marker_pos.shape)
    roi_min = np.maximum(0, np.floor(all_marker_pos.reshape(-1,2).min(axis=0)).astype(int) - 10)
    roi_max = np.ceil(all_marker_pos.reshape(-1,2).max(axis=0)).astype(int) + 10
    
    for frame_idx, frame in enumerate(iio.imiter(fn, plugin="pyav")):
        
        t0 = time.time()
                    
        frame = skimage.color.rgb2gray(frame)   
        t1 = time.time()
        
        markers = np.asarray(all_markers[frame_idx]['checked_markers'])
        blocks = np.asarray([blk[1] for blk in all_markers[frame_idx]['blocks']])  # [block_num, 4]
        
        # # test rotate:
        # frame = np.rot90(frame, k=2)
        # markers = (markers - np.array((frame.shape[1], frame.shape[0]))/2) @ 
        # np.array([[-1,0], [0,-1]]) + np.array((frame.shape[1], frame.shape[0]))/2
        # # done


        # print(markers.shape)
        # print(blocks)
        # exit()
        
        label_0_char, label_1_char, blk_dir = infer_image(model, frame, markers, blocks, confidence_thr=0.5)
        # print(label_0_char)
        # print(label_1_char)
        # print(blk_dir)
        # exit()  

        t2 = time.time()
                
        block_labels = []
        for blk_idx, blk in enumerate(blocks):
            label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
            label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]
            direction = blk_dir[blk_idx][0]
            dir_confidence = blk_dir[blk_idx][1]
            
            block_labels.append({'label':label, 'label_confidence': float(label_confidence),
                                 'direction': int(direction), 'dir_confidence': float(dir_confidence)})
        all_markers[frame_idx]['block_labels'] = block_labels
                
        frame_marker = frame.copy()
        frame1_marker = frame.copy()
        frame1_marker_black = np.zeros_like(frame)
        frame2_marker = frame.copy()
        # print(len(block_candidates))
        draw_block_code(frame1_marker, markers, blocks, label_0_char, label_1_char, blk_dir, wrap_text=False)
        draw_block_code(frame1_marker_black, markers, blocks, label_0_char, label_1_char, blk_dir, wrap_text=True)
        draw_block_corners(frame2_marker, markers, blocks, label_0_char, label_1_char, blk_dir)
        
        block_pts = [markers[block_indices].reshape(-1,1,2) for block_indices in blocks]
        cv2.polylines(frame_marker,block_pts,True, 0.6, thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(frame1_marker,block_pts,True, 0.6, thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(frame1_marker_black,block_pts,True, 0.6, thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(frame2_marker,block_pts,True, 0.6, thickness=1, lineType=cv2.LINE_AA)
        
        frame_marker = utils.draw_markers(frame_marker, markers, color=0.5+0.5)
        frame1_marker = utils.draw_markers(frame1_marker, markers, color=0.5+0.5)
        frame1_marker_black = utils.draw_markers(frame1_marker_black, markers, color=0.5+0.5)
        
        t3 = time.time()
        
        h,w = frame.shape[:2]
        frame_marker = np.concatenate((
            frame_marker[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]], #[:,w//2:], 
            frame1_marker_black[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]], #[:,w//2:],
        ), axis=1)
        
        frame_marker = skimage.color.gray2rgb(frame_marker)
        t4 = time.time()
        
        # print(frame_marker.shape)
        
        if out_fn is not None and out_file is None:
            out_file = cv2.VideoWriter(out_fn, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_marker.shape[1], frame_marker.shape[0]))
           
            out_img = (frame_marker*255).astype(np.uint8)
            out_file.write(out_img)
                
        t5 = time.time()
        
        if frame_idx % 500 == 0:
            print(frame_idx,  f' Infer_image: {t2 - t1:3f}', f' Total: {t5 - t0:3f}' )
                    
    if out_file is not None:
        out_file.release()
        
    if marker_out_fn is None or len(marker_out_fn) == 0:
        marker_out_fn = marker_fn
        
    if marker_out_fn is not None and len(marker_out_fn) > 0:
        with open(marker_out_fn, 'w') as f:
            json.dump(all_markers, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-model', default=os.path.join('ckpts/0520/block', 'checkpoint_200.pt'), type=str)
    parser.add_argument('-c', '--cam-idx', type=str, required=True)
    parser.add_argument('--output-marker-file', type=str, default='')
    parser.add_argument('--dataset-folder', type=str, default=os.path.join(''))
    parser.add_argument('--dataset-def', type=str, default='dataset/mocap0520/label_files.json')
    
    
    args = parser.parse_args()  
    args.input_video = f"dataset/mocap0517/origin/{args.cam_idx}.avi"  
    args.marker_file = f"dataset/mocap0517/result/{args.cam_idx}_r.json"
    args.output_video = f"dataset/mocap0517/result/{args.cam_idx}_block.mp4"


    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # print(args.dataset_folder, args.dataset_def)
    dataset = BlockCodeDataset(args.dataset_folder, args.dataset_def, size=128*500, train=True)

    model = BlockNet(label0_chars=len(dataset.label_characters_0), label1_chars=len(dataset.label_characters_1), blk_dirs=5).to(device)
    
    with open(args.block_model, 'rb') as f:
        data = torch.load(f, map_location=device, weights_only=True)
        
    model.load_state_dict(data['model'])
    model.eval()
    
    video_fn = args.input_video
    out_fn = args.output_video
    marker_fn = args.marker_file
    marker_out_fn = args.output_marker_file
        
    if len(out_fn) == 0:
        out_fn = os.path.splitext(video_fn)
        out_fn = out_fn[0] + '_block.mp4'
        
    if len(marker_out_fn) == 0:
        marker_out_fn = marker_fn[:-5] + '_block.json'

    print(video_fn)
    print(out_fn)
    print(marker_out_fn)

    test_video(model, fn = video_fn, out_fn=out_fn, marker_fn=marker_fn, marker_out_fn=marker_out_fn)
    