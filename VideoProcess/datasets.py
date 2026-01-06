import scipy.spatial
import skimage
import os
import numpy as np
import scipy
import torch
import torchvision.transforms.v2 as torch_trans
import json

import utils
import matplotlib.pyplot as plt
import cv2
import itertools
import random

def load_data_1(dataset_folder):
    with open(os.path.join(dataset_folder, 'annotations', 'default.json'), 'r') as f:
        data_json = json.load(f)
            
    data = [{
        'img_path': item['image']['path'],
        'img_size': item['image']['size'],
        'markers': np.array([anno['points'] for anno in item['annotations'] if anno['type'] == 'points']),
        } for item in data_json['items'] if len(item['annotations']) > 0]
        
    
    for item in data:            
        item['marker_bb'] = np.array((item['markers'].min(axis=0), item['markers'].max(axis=0)))
                  
    for item in data:  
        fn = os.path.join(dataset_folder, 'images', 'default', item['img_path'])
        img = skimage.io.imread(fn)
        img = skimage.color.rgb2gray(img)
        item['img'] = img
        
    print(item['img_size'])
    print(item['img'].shape)
    
    return data

def load_data_mine(dataset_folder, label_file):
    with open(os.path.join(dataset_folder, label_file), 'r') as f:
        data_json = json.load(f)
        
    if 'label_files' in data_json:
        data_json_new = []  
        for fn_info in data_json['label_files']:
            sub_path = os.path.join(*fn_info['folder'])
            fn = os.path.join(sub_path, fn_info['label_file'])
            with open(os.path.join(dataset_folder, fn), 'r') as f:
                sub_data_json = json.load(f)
                for item in sub_data_json:
                    item['image']['path'] = os.path.join(sub_path, item['image']['path'])
                data_json_new.extend(sub_data_json)
                # print(fn, len(sub_data_json), len(data_json_new))
        data_json = data_json_new

    def parse_blocks(markers, blocks):
        markers = np.asarray(markers)
        black_blocks = []
        number_blocks = []
        for blk in blocks:
            corners = blk['corners']
            label = blk['label']
            direction = blk['direction']
            
            marker_dist = scipy.spatial.distance_matrix(markers, corners)
            marker_idx = np.argmin(marker_dist, axis=0)
            corners = tuple(marker_idx.tolist())  # xy -> idx
            
            blk_def = {
                'corners': corners, 
                'corners_sorted': sorted(corners),
                'label': label,
                'direction': direction,
            }
            
            if label == '*':
                black_blocks.append(blk_def)
            else:
                number_blocks.append(blk_def)
        
        return black_blocks, number_blocks
    
    data = []
    label_characters_0 = set()
    label_characters_1 = set()
    for item in data_json:
        frame = {
            'img_path': item['image']['path'],
            'img_size': item['image']['size'],
            'markers': np.array(item['keypoints']),
        }
        frame['marker_bb'] = np.array((frame['markers'].min(axis=0), frame['markers'].max(axis=0)))
        frame['marker_distances'] = scipy.spatial.distance_matrix(frame['markers'], frame['markers'])
        frame['marker_distance_order'] = np.array([np.argsort(row) for row in frame['marker_distances']])
        
        black_blocks, number_blocks = parse_blocks(item['keypoints'], item['blocks'])
        frame['black_blocks'] = black_blocks
        frame['number_blocks'] = number_blocks
        
        frame['marker_to_block_map'] = [[] for _ in frame['markers']]
        marker_neighbors = [set() for _ in frame['markers']]
        for blk in itertools.chain(black_blocks, number_blocks):
            label = blk['label']
            if label != '*':
                label_characters_0.add(label[0])
                label_characters_1.add(label[1])
            
            corners = blk['corners']
            for marker_order, marker_idx in enumerate(corners):
                frame['marker_to_block_map'][marker_idx].append((blk, marker_order))
                marker_neighbors[marker_idx].add(corners[(marker_order + 1) % len(corners)])
                marker_neighbors[marker_idx].add(corners[marker_order - 1])
                
        frame['marker_neighbors'] = [list(neighbor) for neighbor in marker_neighbors]
        
        data.append(frame)
        
    label_characters_0 = sorted(list(label_characters_0)) + ['*', '-']
    label_characters_1 = sorted(list(label_characters_1)) + ['*', '-']
                      
    for item in data:
        img_path = item["img_path"].replace('\\', '/')
        fn = os.path.join(dataset_folder, img_path)
        img = skimage.io.imread(fn)
        img = skimage.color.rgb2gray(img)
        item['img'] = img
        
    print(item['img_size'])
    print(item['img'].shape)
    print(f'label_characters: {label_characters_0, label_characters_1}')
    
    return data, label_characters_0, label_characters_1


class ImageAugmenter:
    def __init__(self) -> None:
        jitter = torch_trans.ColorJitter(brightness=0.2, hue=0.1, contrast=0.1, saturation=0.1)
        blurrer = torch_trans.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 3.))
        posterizer = torch_trans.RandomPosterize(bits=2)
        sharpness_adjuster = torch_trans.RandomAdjustSharpness(sharpness_factor=2)
        autocontraster = torch_trans.RandomAutocontrast()
        equalizer = torch_trans.RandomEqualize()
        jpeg = torch_trans.JPEG((5, 50))
        
        self.all_transform = [
            jitter, 
            blurrer, 
            # posterizer, 
            sharpness_adjuster, 
            # autocontraster, 
            # equalizer, 
            # jpeg
        ]
        
    def apply(self, image:torch.Tensor):
        trans = random.choices(self.all_transform, k=random.randint(1, len(self.all_transform)))
                
        image = (image*255).type(torch.uint8)
        for idx in torch.randperm(len(trans)):
            transform = trans[idx]
            image = transform(image)
            
        image = image.type(torch.get_default_dtype()) / 255
        return image


class MarkerDataset:
    def __init__(self, dataset_folder='add', dataset_file:str='labels.json', size=12800,
                 block_size=64, margin=50, train=None, max_num_markers=1, 
                 output_mask_image=False, output_line_mask_image=False,
                 augment_image=False, debugging=None
                ):
        # self.data = load_data_1(dataset_folder)
        self.data, self.label_characters_0, self.label_characters_1 = load_data_mine(dataset_folder, dataset_file)
        # self.block_size = 27
        # self.margin = 20
        self.block_size = block_size
        self.margin = margin
        self.half_block_size = self.block_size // 2
        self.safe_margin = self.half_block_size + self.margin + 1

        self.size = size
        self.train = train
        self.max_num_markers = max_num_markers
        self.output_mask_image = output_mask_image
        self.output_line_mask_image = output_line_mask_image
        self.debugging = debugging
        self.augment_image = augment_image
        
        self.augmenter = ImageAugmenter()
                    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if self.train is None:
            item = np.random.choice(self.data)
        elif self.train:
            item = np.random.choice(self.data[1:])
        else:
            item = self.data[0]
            
        height, width = item['img_size']        
        
        while True:
            marker_bb = np.array([[self.safe_margin, self.safe_margin], [width - self.safe_margin, height - self.safe_margin]])
            if np.random.rand() > 0.5:
                # marker_bb[0] = np.maximum(item['marker_bb'][0], marker_bb[0])
                # marker_bb[1] = np.minimum(item['marker_bb'][1], marker_bb[1])
                ref_marker = item['markers'][np.random.randint(item['markers'].shape[0])]
                
                ref_block_size = self.block_size
                r_w = int(np.random.rand() * ref_block_size)
                r_h = int(np.random.rand() * ref_block_size)
                
                marker_bb[0] = np.maximum(ref_marker - (r_h, r_w), marker_bb[0])
                marker_bb[1] = np.minimum(ref_marker - (r_h, r_w) + ref_block_size, marker_bb[1])
                
            ctr = np.random.rand(2)*(marker_bb[1] - marker_bb[0]) + marker_bb[0]
            ctr = np.round(ctr).astype(int)
            
            btm_left = ctr - [self.safe_margin - 1, self.safe_margin - 1]
            top_right = btm_left + (self.block_size + self.margin * 2 + 1)
            img = item['img'][btm_left[1]:top_right[1], btm_left[0]:top_right[0]]
            
            if img.size == (self.block_size + self.margin * 2 + 1)**2:
                break
            
            # print(img.shape, img.size, (self.block_size + self.margin * 2 + 1))
        
        
        if self.augment_image and self.augmenter is not None:
            with torch.no_grad():
                img = self.augmenter.apply(torch.from_numpy(img).view(1, img.shape[0], img.shape[1])).numpy().reshape(img.shape)
        
        if self.train is not None and not self.train:
            trans, dst = utils.random_perspective(img.shape[1], img.shape[0])
        else:
            trans, dst = utils.random_perspective(img.shape[1], img.shape[0], distortion_scale=0.8*0.1, scale=[0.95, 1.1], rotation=[-180, 180], flip=0.5)
        img_t = skimage.transform.warp(img, trans.inverse, order=3)
        
        marker_candidates = [(i,m) for i, m in enumerate(item['markers']) if (np.maximum(m, btm_left) == m).all() and (np.minimum(m, top_right) == m).all()]
        
        if len(marker_candidates) > 0:
            marker_ids, markers = zip(*marker_candidates)
        else:
            marker_ids, markers = [], []
        
        if len(markers) > 0:
            markers = np.asarray(markers) - btm_left
            markers = skimage.transform.matrix_transform(markers, trans.params)
            
        block_0 = (img.shape[1] - self.block_size) // 2        
        block_ctr = block_0 + self.half_block_size
        block = img_t[block_0:block_0+self.block_size, block_0:block_0+self.block_size]
        
        marker_in_block_candidates = [(marker_ids[i], m-block_0) for i,m in enumerate(markers) 
                                                        if abs(m[0] - block_ctr) <= self.half_block_size and abs(m[1]-block_ctr) <= self.half_block_size]
        if len(marker_in_block_candidates) > 0:
            marker_ids_in_blocks, markers_in_blocks = zip(*marker_in_block_candidates)
        else:
            marker_ids_in_blocks, markers_in_blocks = [], []
            
        
        the_marker = None
        if len(markers_in_blocks) > 0:
            dist = [ (abs(m[0] - block_ctr) + abs(m[1]-block_ctr))
                    for m in markers if abs(m[0] - block_ctr) <= self.half_block_size and abs(m[1]-block_ctr) <= self.half_block_size]
            the_marker = markers_in_blocks[np.argmin(dist)]
                                
        if self.debugging:
            def debug():
                img_m = utils.draw_markers(img_t, markers)        
                block_m = block
                if len(markers) > 0:
                    block_m = utils.draw_markers(block, markers-block_0, radius=1, color=1)
                    block_m = utils.draw_markers(block_m, markers_in_blocks, radius=2, color=0.5)
                    if the_marker is not None:
                        block_m = utils.draw_markers(block_m, [the_marker], radius=2, color=1)
                
                plt.subplot(2, 2, 1)
                plt.imshow(img_m, cmap=plt.cm.gray)
                plt.gca().add_patch(plt.Rectangle([(img.shape[1] - self.block_size) // 2, (img.shape[0] - self.block_size)//2], self.block_size, self.block_size, edgecolor = 'pink', fill=False, lw=3))
                plt.subplot(2, 2, 2)
                # plt.imshow(img, cmap=plt.cm.gray)
                plt.imshow(img_t, cmap=plt.cm.gray)
                plt.gca().add_patch(plt.Rectangle([(img.shape[1] - self.block_size) // 2, (img.shape[0] - self.block_size)//2], self.block_size, self.block_size, edgecolor = 'pink', fill=False, lw=3))
                
                plt.subplot(2, 2, 3)
                plt.imshow(block, cmap=plt.cm.gray)
                plt.subplot(2, 2, 4)
                plt.imshow(block_m, cmap=plt.cm.gray)
                
                # plt.show()
                plt.savefig("debug.png")
            debug()
        
        if self.max_num_markers <= 1:
            output = [0, -1, -1] if the_marker is None else [1, the_marker[0], the_marker[1]]
                
        else:
            output = np.array([[0, -1, -1]]*self.max_num_markers)
            num = min(self.max_num_markers, len(markers_in_blocks))
            for i, mk in enumerate(markers_in_blocks[:num]):
                output[i] = [1, mk[0], mk[1]]
                            
        if self.output_mask_image:
            output_img = np.zeros_like(img_t)
            output = [output] if isinstance(output, list) else output
            output = np.round(output, 0).astype(int)
            
            has_marker = False
            for mk in output:
                if mk[0] == 0:
                    break                
                output_img[mk[2] + block_0, mk[1] + block_0] = 1
                has_marker = True
                
            # output_img = cv2.GaussianBlur(output_img, (0, 0), 5, borderType=cv2.BORDER_REPLICATE  )
            output_img = skimage.filters.gaussian(output_img, sigma=2)
            
            if has_marker:
                output_img = output_img / output_img.max()
            
            output_img = output_img[block_0:block_0+self.block_size, block_0:block_0+self.block_size]
            
            if self.debugging:
                # print(output)
                plt.imshow(np.minimum(block + output_img, 1))
                # plt.show()
                
            output = output_img.reshape(1, self.block_size, self.block_size)
            # print("output1.shape:", output.shape)
            
        if self.output_mask_image and self.output_line_mask_image:
            neighbor_markers = [item['markers'][item['marker_neighbors'][m_idx]] for m_idx in marker_ids_in_blocks]
            
            output_line_img = np.zeros_like(img_t)
            
            has_neighbors = False
            for mk, neighbors in zip(markers_in_blocks, neighbor_markers):                
                if len(neighbors) == 0:
                    continue
                
                neighbors = np.asarray(neighbors) - btm_left
                neighbors = skimage.transform.matrix_transform(neighbors, trans.params)
                neighbors = np.round(neighbors, 0).astype(int)
                mk = np.round(mk, 0).astype(int)
                
                for nb in neighbors:
                    output_line_img = cv2.line(output_line_img, mk + block_0, nb, color=1)
                    
                has_neighbors = True
                        
            # output_line_img = cv2.GaussianBlur(output_line_img, (0, 0), 5, borderType=cv2.BORDER_REPLICATE  )
            output_line_img = skimage.filters.gaussian(output_line_img, sigma=2)
            
            if has_neighbors:
                output_line_img = output_line_img / output_line_img.max()
            
            output_line_img = output_line_img[block_0:block_0+self.block_size, block_0:block_0+self.block_size]
                
            if self.debugging:
                plt.imshow(np.minimum(block + output_line_img, 1))
                            
            output = np.stack((output_img, output_line_img), axis=0)
        
                    
        block = torch.from_numpy(block.astype(np.float32)).reshape(1, self.block_size, self.block_size)
        output = torch.as_tensor(output, dtype=torch.float32)
        return block, output
    

class BlackBlockIdentifyDataset:
    def __init__(self, dataset_folder='add', 
                 dataset_file:str='labels.json',
                 size=12800,
                 train=None,
                 augment_image=False,
                 debugging=None
                ):
        # self.data = load_data_1(dataset_folder)
        self.data, self.label_characters_0, self.label_characters_1 = load_data_mine(dataset_folder, dataset_file)
        
        # self.block_size = 27
        # self.margin = 20
        self.block_size = 60
        self.margin = 30
        self.half_block_size = self.block_size // 2
        self.safe_margin = self.half_block_size + self.margin + 1
        
        self.point_distance_thr = 60
        
        self.image_size = 64
        self.image_margin = 10
        
        for frame in self.data:
            frame['marker_distances_mask'] = frame['marker_distances'] < self.point_distance_thr
            frame['marker_candidate_neighbor_cnt'] = frame['marker_distances_mask'].sum(axis=-1)

        self.size = size
        self.train = train
        self.debugging = debugging
        self.augment_image = augment_image
                            
        self.augmenter = ImageAugmenter()
        
    def __len__(self):
        return self.size
            
    def __getitem__(self, idx):
        if self.train is None:
            item = np.random.choice(self.data)
        elif self.train:
            item = np.random.choice(self.data[1:])
        else:
            item = self.data[0]
            
        height, width = item['img_size']
        black_blocks = item['black_blocks']
        markers = item['markers']
        image = item['img']
        
        is_black_block = False
        
        prob = np.random.rand()
        # we need four point
        if prob < 0.2: 
            # gt block
            is_black_block = True
            block = np.random.choice(black_blocks)
            corners = markers[list(block['corners'])]
            
        elif prob < 0.9:
            # random markers
            while True:
                marker_idx = np.random.randint(len(markers))
                num_candidates = item['marker_candidate_neighbor_cnt'][marker_idx]
                if num_candidates < 4:
                    continue
                
                corners_idx = np.random.choice(item['marker_distance_order'][marker_idx, :num_candidates], size=4, replace=False)
                corners = markers[corners_idx]
                order = utils.check_convex_4_points(corners)
                if order[0] == -1:
                    continue
        
                corners_idx = np.asarray(corners_idx)[order]
                corners = corners[order]
                break
            
            corners_idx_sorted = sorted(corners_idx)
            for blk in black_blocks:
                if corners_idx_sorted == blk['corners_sorted']:
                    is_black_block = True
                    break
            
        else:
            marker_bb = np.array([[self.safe_margin, self.safe_margin], [width - self.safe_margin, height - self.safe_margin]])            
            while True:
                ctr = np.random.rand(2)*(marker_bb[1] - marker_bb[0]) + marker_bb[0]
                offsets = np.random.uniform(-self.point_distance_thr, self.point_distance_thr, size=(4, 2))*0.8
                
                corners = ctr.reshape(1, 2) + offsets
                corners = np.round(corners).astype(int)
                order = utils.check_convex_4_points(corners)
                if order[0] == -1:
                    continue
                
                corners = corners[order]
                break
        
        
        # further perturb it
        corners = corners + np.random.uniform(-3, 3, size=(4, 2))
        corners = np.roll(corners, shift=np.random.randint(4), axis=0)
        
        margin = 15
        xy_min = np.maximum(0, np.floor(corners.min(axis=0)).astype(int) - margin)
        xy_max = np.minimum((width - 1, height - 1), np.ceil(corners.max(axis=0)).astype(int) + margin)
        image_block = image[xy_min[1]:xy_max[1]+1, xy_min[0]:xy_max[0]+1]
        
        if self.augment_image and self.augmenter is not None:
            with torch.no_grad():
                image_block = self.augmenter.apply(torch.from_numpy(image_block).view(1,image_block.shape[0],image_block.shape[1])).numpy().reshape(image_block.shape)
                
        # Set up the destination points for the perspective transform
        blk_min = self.image_margin
        blk_max = self.image_size - self.image_margin
        dst = np.array([
            [blk_min, blk_min],
            [blk_max - 1, blk_min],
            [blk_max - 1, blk_max - 1],
            [blk_min, blk_max - 1]], dtype=np.float32)

        # Calculate the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform((corners - xy_min).astype(np.float32).reshape(-1,2), dst)
        # warped = cv2.warpPerspective(image_block.swapaxes(0,1), M, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST).swapaxes(0,1)
        warped = cv2.warpPerspective(image_block, M, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST)
        
        
        block = torch.from_numpy(warped.astype(np.float32)).reshape(1, self.image_size, self.image_size)
        output = torch.as_tensor(1.0 if is_black_block else 0.0, dtype=torch.float32)
                
        return block, output


class BlockCodeDataset:
    def __init__(self, dataset_folder='add', 
                 dataset_file:str='labels.json',
                 size=12800,
                 train=None,
                 augment_image=False,
                 debugging=None
                ):
        # self.data = load_data_1(dataset_folder)
        self.data, self.label_characters_0, self.label_characters_1 = load_data_mine(dataset_folder, dataset_file)
        
        # self.block_size = 27
        # self.margin = 20
        self.block_size = 60
        self.margin = 30
        self.half_block_size = self.block_size // 2
        self.safe_margin = self.half_block_size + self.margin + 1
        
        self.point_distance_thr = 60
        
        self.block_margin = 15
        self.image_size = 64
        self.image_margin = 10
        
        # self.margin = 30*2
        # self.block_margin = 30
        # self.image_size = 128
        # self.image_margin = 40
        
        for frame in self.data:
            frame['marker_distances_mask'] = frame['marker_distances'] < self.point_distance_thr
            frame['marker_candidate_neighbor_cnt'] = frame['marker_distances_mask'].sum(axis=-1)
            
        self.size = size
        self.train = train
        self.debugging = debugging
        self.augment_image = augment_image
        
        self.augmenter = ImageAugmenter()
                    
    def __len__(self):
        return self.size
            
    def __getitem__(self, idx):
        if self.train is None:
            item = np.random.choice(self.data)
        elif self.train:
            item = np.random.choice(self.data[1:])
        else:
            item = self.data[0]
            
        height, width = item['img_size']
        black_blocks = item['black_blocks']
        number_blocks = item['number_blocks']
        markers = item['markers']
        image = item['img']
        
        block_label = '--'
        block_dir = 4
        
        prob = np.random.rand()
        # we need four point
        if prob < 0.2: 
            # gt block
            block = np.random.choice(black_blocks)
            block_label = '**'
            block_dir = 4
            corners = markers[list(block['corners'])]
            
        elif prob < 0.7:
            # gt number
            block = np.random.choice(number_blocks)
            block_label = block['label']
            block_dir = block['direction']
            corners = markers[list(block['corners'])]
            
        elif prob < 0.9:
            # random markers
            while True:
                marker_idx = np.random.randint(len(markers))
                num_candidates = item['marker_candidate_neighbor_cnt'][marker_idx]
                if num_candidates < 4:
                    continue
                
                corners_idx = np.random.choice(item['marker_distance_order'][marker_idx, :num_candidates], size=4, replace=False)
                corners = markers[corners_idx]
                order = utils.check_convex_4_points(corners)
                if order[0] == -1:
                    continue
        
                corners_idx = np.asarray(corners_idx)[order]
                corners = corners[order]
                break
            
            corners_idx_sorted = sorted(corners_idx)
            for blk in black_blocks:
                if corners_idx_sorted == blk['corners_sorted']:
                    block_label = '**'
                    block_dir = 4
                    break
                
            for blk in number_blocks:
                if corners_idx_sorted == blk['corners_sorted']:
                    block_label = blk['label']
                    block_dir = blk['direction']
                    break
            
        else: #random point
            marker_bb = np.array([[self.safe_margin, self.safe_margin], [width - self.safe_margin, height - self.safe_margin]])            
            while True:
                ctr = np.random.rand(2)*(marker_bb[1] - marker_bb[0]) + marker_bb[0]
                offsets = np.random.uniform(-self.point_distance_thr, self.point_distance_thr, size=(4, 2))*0.8
                
                corners = ctr.reshape(1, 2) + offsets
                corners = np.round(corners).astype(int)
                order = utils.check_convex_4_points(corners)
                if order[0] == -1:
                    continue
                
                corners = corners[order]
                break
        
        
        # further perturb it
        corners = corners + np.random.uniform(-3, 3, size=(4, 2))
        dir_shift = np.random.randint(4)
        corners = np.roll(corners, shift=dir_shift, axis=0)
        if block_dir < 4:
            block_dir = (block_dir+dir_shift) % 4
        
        wrapped = self.extract_block(image, corners, do_augment=self.augment_image)
        
        
        block = torch.from_numpy(wrapped.astype(np.float32)).reshape(1, self.image_size, self.image_size)
        out_label_0 = torch.zeros(len(self.label_characters_0))
        out_label_1 = torch.zeros(len(self.label_characters_1))
        out_dir = torch.zeros(5)
        out_label_0[self.label_characters_0.index(block_label[0])] = 1
        out_label_1[self.label_characters_1.index(block_label[1])] = 1
        out_dir[block_dir] = 1
                
        return block, (out_label_0, out_label_1, out_dir)
    
    def extract_block(self, image:np.ndarray, corners:np.ndarray, do_augment:bool=False):
        
        height, width = image.shape[:2]
        
        margin = self.block_margin
        xy_min = np.maximum(0, np.floor(corners.min(axis=0)).astype(int) - margin)
        xy_max = np.minimum((width - 1, height - 1), np.ceil(corners.max(axis=0)).astype(int) + margin)
        image_block = image[xy_min[1]:xy_max[1]+1, xy_min[0]:xy_max[0]+1]
        
        
        if do_augment and self.augmenter is not None:
            with torch.no_grad():
                image_block = self.augmenter.apply(torch.from_numpy(image_block).view(1,image_block.shape[0],image_block.shape[1])).numpy().reshape(image_block.shape)
                
        # Set up the destination points for the perspective transform
        blk_min = self.image_margin
        blk_max = self.image_size - self.image_margin
        dst = np.array([
            [blk_min, blk_min],
            [blk_max - 1, blk_min],
            [blk_max - 1, blk_max - 1],
            [blk_min, blk_max - 1]], dtype=np.float32)

        # Calculate the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform((corners - xy_min).astype(np.float32).reshape(-1,2), dst)
        # wrapped = cv2.warpPerspective(image_block.swapaxes(0,1), M, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST).swapaxes(0,1)
        wrapped = cv2.warpPerspective(image_block, M, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST)
        
        return wrapped
    
    def code_index(self, label:str, label_2:str|None=None):
        if label is None or len(label) == 0:
            return -1
        
        if label_2 is None and len(label) < 2:
            return -1
        
        if label[0] == '*':
            return -2
        
        if label[0] == '-':
            return -3
        
        label_1 = label[0]
        if label_2 is None:
            label_2 = label[1]
            
        try:
            idx_1 = self.label_characters_0.index(label_1)
            idx_2 = self.label_characters_1.index(label_2)
        except IndexError:
            return -1
        
        return idx_1*(len(self.label_characters_1)-2) + idx_2
    
    
class EdgeDataset:
    def __init__(self, dataset_folder='add', dataset_file:str='labels.json',
                 size=12800, train=None, augment_image=False, debugging=None):
        # self.data = load_data_1(dataset_folder)
        self.data, self.label_characters_0, self.label_characters_1 = load_data_mine(dataset_folder, dataset_file)
        
        self.safe_margin = 60
        
        self.image_size = 64
        self.image_margin = 10
        
        self.point_distance_thr = 60
        
        for frame in self.data:
            frame['marker_distances_mask'] = frame['marker_distances'] < self.point_distance_thr
            frame['marker_candidate_neighbor_cnt'] = frame['marker_distances_mask'].sum(axis=-1)
            close_markers = np.nonzero(frame['marker_distances_mask'])
            markers_in_range = [set() for i in range(len(frame['markers']))]
            for idx_0, idx_1 in zip(*close_markers):
                if idx_0 == idx_1:
                    continue
                markers_in_range[idx_0].add(idx_1)
                markers_in_range[idx_1].add(idx_0)
                
            frame['markers_in_range'] = [list(indices) for indices in markers_in_range]

        self.size = size
        self.train = train
        self.debugging = debugging
        self.augment_image = augment_image
        
        self.augmenter = ImageAugmenter()
                    
    def __len__(self):
        return self.size
            
    def __getitem__(self, idx):
        if self.train is None:
            item = np.random.choice(self.data)
        elif self.train:
            item = np.random.choice(self.data[1:])
        else:
            item = self.data[0]
            
        height, width = item['img_size']
        markers = item['markers']
        image = item['img']
        marker_neighbors = item['marker_neighbors']
        markers_in_range = item['markers_in_range']
        marker_to_block_map = item['marker_to_block_map']
        
        is_connected = False
        
        marker_bb = np.array([[self.safe_margin, self.safe_margin], [width - self.safe_margin, height - self.safe_margin]])     
        prob = np.random.rand()
        # we need four point
        if prob < 0.95:  # random markers
            idx_0, idx_1 = np.random.choice(list(range(len(markers))), 2, replace=False)
            if prob < 0.4 and len(marker_neighbors[idx_0]) > 0:
                idx_1 = random.choice(marker_neighbors[idx_0])
            elif prob < 0.55 and len(marker_to_block_map[idx_0]):
                block, idx_0_in_block = random.choice(marker_to_block_map[idx_0])
                blk_corner:list = block['corners']
                idx_1 = blk_corner[(idx_0_in_block + 2) % 4] # diag                 
                # assert idx_0_in_block == blk_corner.index(idx_0)
                # print('here')         
            elif prob < 0.85 and len(markers_in_range[idx_0]) > 0:
                idx_1 = np.random.choice(markers_in_range[idx_0])
                
            if idx_1 in marker_neighbors[idx_0]:
                is_connected = True
                
            edge_ends = markers[[idx_0, idx_1], :]
            
            if prob > 0.9:
                edge_ends[np.random.randint(0,1)] = np.random.rand(2)*(marker_bb[1] - marker_bb[0]) + marker_bb[0]
                is_connected = False                
            
        else: #random point
            edge_ends = np.random.rand(2,2)*(marker_bb[1] - marker_bb[0]).reshape(1,2) + marker_bb[0].reshape(1,2)
        
        
        # further perturb it
        edge_ends = edge_ends + np.random.uniform(-3, 3, size=(2, 2))
        
        wrapped = self.extract_block(image, edge_ends, do_augment=self.augment_image)        
        
        block = torch.from_numpy(wrapped.astype(np.float32)).reshape(1, wrapped.shape[0], wrapped.shape[1])
        out_label = torch.tensor([1.]) if is_connected else torch.tensor([0.])
                
        return block, out_label
    
    def extract_block(self, image:np.ndarray, edge_ends:np.ndarray, do_augment:bool=False):
        wrapped = EdgeDataset.extract_block_(image=image,
                                  block_image_size=self.image_size,
                                  block_image_margin=self.image_margin,
                                  edge_ends=edge_ends, 
                                  do_augment=do_augment, augmenter=self.augmenter,
                                  debugging=self.debugging
                                  )
        
        return wrapped
    
    @staticmethod
    def extract_block_(image:np.ndarray, edge_ends:np.ndarray, 
                      block_image_size:int, block_image_margin:int,
                      do_augment:bool=False, 
                      augmenter:ImageAugmenter|None=None, 
                      debugging:bool=False):
        height, width = image.shape[:2]
        
        edge = edge_ends[1] - edge_ends[0]
        edge_length = np.linalg.norm(edge)
        edge_dir = edge / edge_length
        perp_dir = np.array([-edge_dir[1], edge_dir[0]])
        
        corners = np.array([
            edge_ends[0] - perp_dir*(edge_length*0.5),
            edge_ends[1] - perp_dir*(edge_length*0.5),
            edge_ends[1] + perp_dir*(edge_length*0.5),
            edge_ends[0] + perp_dir*(edge_length*0.5),
        ])
        
        margin = 30
        xy_min = np.maximum(0, np.floor(corners.min(axis=0)).astype(int) - margin)
        xy_max = np.minimum((width - 1, height - 1), np.ceil(corners.max(axis=0)).astype(int) + margin)
        image_block = image[xy_min[1]:xy_max[1]+1, xy_min[0]:xy_max[0]+1]
        
        if debugging:
            def debug_draw():
                plt.imshow(image_block)
                corner_xy = (corners - xy_min).astype(np.float32)
                corner_xy = np.concatenate((corner_xy, corner_xy[:1]), axis=0)
                edge_ends_xy = edge_ends-xy_min
                plt.plot(corner_xy[:,0], corner_xy[:,1])
                plt.plot(edge_ends_xy[:,0], edge_ends_xy[:,1])
                plt.show()
            debug_draw()        
        
        if do_augment and augmenter is not None:
            with torch.no_grad():
                image_block = augmenter.apply(torch.from_numpy(image_block).view(1,image_block.shape[0],image_block.shape[1])).numpy().reshape(image_block.shape)
                
        # Set up the destination points for the perspective transform
        blk_x_min = block_image_margin
        blk_x_max = block_image_size - block_image_margin
        blk_y_min = block_image_margin
        blk_y_max = block_image_size - block_image_margin
        dst = np.array([
            [blk_x_min, blk_y_min],
            [blk_x_max - 1, blk_y_min],
            [blk_x_max - 1, blk_y_max - 1],
            [blk_x_min, blk_y_max - 1]], dtype=np.float32)

        # Calculate the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform((corners - xy_min).astype(np.float32).reshape(-1,2), dst)
        wrapped = cv2.warpPerspective(image_block, M, (block_image_size, block_image_size), flags=cv2.INTER_NEAREST)
        
        return wrapped
    
    
def test_marker():
    dataset = MarkerDataset(os.path.join('tests', 'video1Marker', 'labels'), max_num_markers=20, output_mask_image=True, output_line_mask_image=True, debugging=True)
    for i in range(10):
        block, output = dataset[i]
        print(block, output)
        plt.imshow(block.numpy()[0], cmap='gray', vmin=0, vmax=1)
        plt.show()
    
    
def test_block():
    dataset = BlockCodeDataset(os.path.join('tests', 'video1Marker', 'labels'), debugging=True)
    dataset = BlockCodeDataset(os.path.join('tests', 'videoMultiViewMarker0', 'labels'), debugging=True)
    dataset = BlockCodeDataset(os.path.join('tests', 'mocap0902', 'labels'), debugging=True)
    for i in range(10):
        block, (out_label_0, out_label_1, out_dir) = dataset[i]
        print('label0: ', out_label_0)
        print('label1: ', out_label_1)
        print('dir: ', out_dir)
        print(np.nonzero(out_label_0.numpy()))
        print(np.nonzero(out_label_1.numpy()))
        print(dataset.label_characters_0[np.nonzero(out_label_0.numpy())[0][0]], 
            dataset.label_characters_1[np.nonzero(out_label_1.numpy())[0][0]])
        
        dir_idx = np.nonzero(out_dir.numpy().flatten())[0][0]
        print('dir_idx:', dir_idx)
        
        un_rotate = torch.rot90(block, k=dir_idx, dims=[1,2]).numpy()[0]
        
        plt.subplot(1,2,1)
        plt.imshow(block.numpy()[0], cmap='gray', vmin=0, vmax=1)
        plt.subplot(1,2,2)
        plt.imshow(un_rotate, cmap='gray', vmin=0, vmax=1)
        plt.show()

   
def test_edge():
    dataset = EdgeDataset(os.path.join('tests', 'videoMultiViewMarker0', 'labels'), debugging=True)
    
    blk_x_min = dataset.image_margin
    blk_x_max = dataset.image_size - dataset.image_margin
    blk_y_min = dataset.image_margin
    blk_y_max = dataset.image_size - dataset.image_margin
    dst = np.array([
        [blk_x_min, blk_y_min],
        [blk_x_max - 1, blk_y_min],
        [blk_x_max - 1, blk_y_max - 1],
        [blk_x_min, blk_y_max - 1],        
        [blk_x_min, blk_y_min],], dtype=np.float32)
    
    for i in range(10):
        block, out_label = dataset[i]
        print('label: ', out_label)
        plt.imshow(block.numpy()[0], cmap='gray', vmin=0, vmax=1)
        plt.plot(dst[:,0], dst[:,1])
        plt.show()
    
        
if __name__ == '__main__':
    # test_marker()
    test_block()
    # test_edge()