import skimage
import numpy as np

from typing import Union, List, Any, Dict
from collections.abc import Iterable
import numbers

import scipy.spatial
import itertools
import cv2
import textwrap
import json

try:
    import ffmpeg
except ImportError:
    print('Cannot import ffmpeg, you may need to run `pip install ffmpeg-python`, note: not python-ffmpeg')
import subprocess

def draw_markers(img:np.ndarray, markers:np.ndarray, radius=3, thickness=1, color=1, inplace=False):
    h, w = img.shape[:2]
    
    res = np.copy(img) if not inplace else img
    
    if isinstance(color, numbers.Number):
        color = [color] * len(markers)
    elif hasattr(color, '__len__') and callable(getattr(color, '__len__')) and len(color) != len(markers):
        color = [color] * len(markers)
    
    for mark, clr in zip(markers, color):
        r,c = mark[:2]
        r = int(np.round(r))
        c = int(np.round(c))
        for i in range(thickness):
            rr, cc = skimage.draw.circle_perimeter(r, c, radius + i, method='andres')
            rr = np.clip(rr, 0, w - 1)
            cc = np.clip(cc, 0, h - 1)
            res[cc, rr] = clr
            
    return res

def random_perspective(width:int, height:int, distortion_scale:float=None, scale=None, rotation=None, flip=0): 
    half_height = height // 2
    half_width = width // 2
    
    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst = src[:]
    
    if distortion_scale is not None:
        bound_height = int(distortion_scale * half_height) + 1
        bound_width = int(distortion_scale * half_width) + 1
        points = np.random.uniform(0, 1, size=(4, 2)) * [[bound_width, bound_height]]
        points[1:3,0] = width - points[1:3,0]
        points[2:4,1] = height - points[2:4,1]    
        dst = points
    
        ctr = points.mean(axis=0, keepdims=True)
        dst -= ctr
        
    if rotation is not None or scale is not None:
        
        if scale is not None:
            scale = list(scale)
            factor = np.random.uniform(*scale)
            dst *= factor
        
        if rotation is not None:
            rotation = list(rotation)
            rot_ang = np.random.uniform(*rotation)
            rot_ang = np.deg2rad(rot_ang)
            ss = np.sin(rot_ang)
            cs = np.cos(rot_ang)
            
            rot_mat = np.array([[ss, -cs], [cs, ss]])
            dst = dst @ rot_mat.T 
            
    if np.random.uniform() < flip:
        dst[:,0] *= -1
            
    dst += [[half_width, half_height]]
        
    
    prep = skimage.transform.ProjectiveTransform()
    prep.estimate(src, dst)
    return prep, dst

def random_transform(img:np.ndarray, markers:np.ndarray, distortion_scale:float, scale=None, rotation=None, flip=0):  
    
    trans, dst = random_perspective(img.shape[1], img.shape[0], distortion_scale, scale, rotation, flip)
    img = skimage.transform.warp(img, trans.inverse, mode='edge')
    
    markers = skimage.transform.matrix_transform(markers, trans.params)
    markers = markers[:,:2].astype(int)
    
    img = draw_markers(img, dst, 5, 3)
    
    return img, markers

def random_blur(img, gaussian:int, motion:int):
    
    
    return img
    
def check_convex_4_points(points:np.ndarray):
    # ref:https://stackoverflow.com/questions/2122305/convex-hull-of-4-points
    
    # points: (n, 4, 2)
    points = np.asarray(points)
    assert points.shape[-2:] == (4, 2)
    single_quadrilateral = False
    if points.ndim == 2:
        points = points.reshape(1, 4, 2)
        single_quadrilateral = True
        
    A = points[:,0]
    B = points[:,1]
    C = points[:,2]
    D = points[:,3]
        
    triangle_ABC = ((A[:,1]-B[:,1])*C[:,0] + (B[:,0]-A[:,0])*C[:,1] + (A[:,0]*B[:,1]-B[:,0]*A[:,1])) > 0
    triangle_ABD = ((A[:,1]-B[:,1])*D[:,0] + (B[:,0]-A[:,0])*D[:,1] + (A[:,0]*B[:,1]-B[:,0]*A[:,1])) > 0
    triangle_BCD = ((B[:,1]-C[:,1])*D[:,0] + (C[:,0]-B[:,0])*D[:,1] + (B[:,0]*C[:,1]-C[:,0]*B[:,1])) > 0
    triangle_CAD = ((C[:,1]-A[:,1])*D[:,0] + (A[:,0]-C[:,0])*D[:,1] + (C[:,0]*A[:,1]-A[:,0]*C[:,1])) > 0
    
    triangle_ABD = triangle_ABD == triangle_ABC
    triangle_BCD = triangle_BCD == triangle_ABC
    triangle_CAD = triangle_CAD == triangle_ABC
    
    convex_order = np.ones(shape=(points.shape[0], 4), dtype=int) * -1
    convex_order[ triangle_ABD & triangle_BCD &~triangle_CAD] = (0, 1, 2, 3)
    convex_order[ triangle_ABD &~triangle_BCD & triangle_CAD] = (0, 1, 3, 2)
    convex_order[~triangle_ABD & triangle_BCD & triangle_CAD] = (0, 3, 1, 2)
    # roll to make sure that 0 is the first one
    convex_order[~triangle_ABC] = np.roll(np.flip(convex_order[~triangle_ABC], axis=-1), 1, axis=-1)
    
    if single_quadrilateral:
        convex_order = convex_order.flatten()

    return convex_order

def calculate_quadrilateral_area(points:np.ndarray):
    # points: (n, 4, 2)
    points = np.asarray(points)
    assert points.shape[-2:] == (4, 2)
    single_quadrilateral = False
    if points.ndim == 2:
        points = points.reshape(1, 4, 2)
        single_quadrilateral = True
        
    lengths = np.linalg.norm(np.diff(points, axis=1, append=points[:,:1]), axis=-1)
    diag_length = np.linalg.norm(points[:,0,:] - points[:,2,:], axis=-1)
    
    s1 = (lengths[:,0] + lengths[:,1] + diag_length) / 2
    s2 = (lengths[:,2] + lengths[:,3] + diag_length) / 2
    T1 = np.sqrt(s1*np.prod(s1[:,None] - lengths[:,:2], axis=-1)*(s1-diag_length))
    T2 = np.sqrt(s2*np.prod(s2[:,None] - lengths[:,2:], axis=-1)*(s2-diag_length))
    
    areas = T1 + T2
    
    return areas
    
def find_quads(points:np.ndarray, distance_thr:float, min_area:float, max_area:float, max_nearest_neighbors:int=12):
    
    # points: (n, 2)
    points = np.asarray(points).astype(np.float32)
    
    # compute distance
    dists = scipy.spatial.distance.cdist(points, points)
    dist_mask:np.ndarray = dists < distance_thr
    np.fill_diagonal(dist_mask, False)
    
    candidates = []
    check_blocks = set()
    
    # find the points within a distance from each point
    indices = np.arange(points.shape[0])
    for idx in indices:
        neighbors, *_ = dist_mask[idx].nonzero()
        neighbors = neighbors.tolist()
        if len(neighbors) < 3:
            continue
                
        curr_candidates = []
        for comb in itertools.combinations(neighbors, 3):
            comb = [idx] + list(comb)
            comb_sorted = sorted(comb)
            
            if tuple(comb_sorted) in check_blocks:
                continue
            
            check_blocks.add(tuple(comb_sorted))
                        
            hull = cv2.convexHull(points[comb])
            if len(hull) < 4: # concave points is not feasible
                continue
            
            edges = (hull - np.roll(hull, 1, axis=0)).reshape(4,2)
            edge_length = np.linalg.norm(edges, axis=-1)
            inner_angles = -(edges * np.roll(edges, 1, axis=0)).sum(axis=-1).astype(float)
            inner_angles /= edge_length * np.roll(edge_length, 1, axis=0)
            inner_angles = np.arccos(inner_angles)
            
            if (inner_angles.max() > np.pi/4*3) or (inner_angles.min() < np.pi / 4):
                continue
            
            if edge_length.max() / edge_length.min() > 4:
                continue
            
            diagonal_length = np.linalg.norm(hull[:2] - hull[2:], axis=-1).flatten()
            if diagonal_length.max() / diagonal_length.min() > 2:
                continue
                                  
            
            hull_point_dist = abs(hull.reshape(1, -1, 2) - points[comb].reshape(-1, 1, 2)).sum(axis=-1)
            hull_point_order = [hull_point_dist[:,i].argmin() for i in range(4)]
            block_index = [comb[i] for i in hull_point_order]
            
            
            curr_candidates.append(block_index)
            
        candidates.extend(curr_candidates)
        
    candidates = np.asarray(sorted(list(candidates)))
    
    areas = calculate_quadrilateral_area(np.take(points, candidates, axis=0))
    valid_areas = (areas >= min_area) & (areas <= max_area)
    candidates = candidates[valid_areas,:]
    
    print('after area filter:', len(candidates))
    
    return candidates

def debug_show_img(img):
    import cv2
    cv2.imshow('image', img)
    while True:
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        
    cv2.destroyWindow('image')
    

def triangulate_nviews(P, ip):
    """ https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices, shape (n, 3, 4).
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([2*n, 4])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[i*2+0,:] = p[2,:]*x[0] - p[0,:]
        M[i*2+1,:] = p[2,:]*x[1] - p[1,:]
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    X = X[:3] / X[3]
    return X

def triangulate_n_views_error(P, pt2d, est_pt3d):
    # P shape (n, 3, 4)
    # pt2d shape (n, 2)
    # pt3d shape (3,)
    P = np.asarray(P)
    pt2d = np.asarray(pt2d)
    est_pt3d = np.asarray(est_pt3d)
    x_bar = np.einsum('nij,j->ni', P[...,:,:3], est_pt3d) + P[...,:,3]
    x_bar = x_bar[:,:2]/x_bar[:,-1:]
    
    error = np.linalg.norm(x_bar - pt2d, axis=-1)
    
    return error
    

def triangulate_robust(P, pts_2d, inliner_thr):
    P = np.asarray(P) # shape (n, 3, 4)
    pts_2d = np.asarray(pts_2d)[:,:2] # shape (n, 2)
    
    checked_inliner_set = set()
    best_fit_error = 1e20
    best_inliners = None
    best_X = None
    best_reproj_error = None
    
    # using the same idea of RANSAC but, instead of using random points,
    # we try to check all combinations
    for cam_i, cam_j in itertools.combinations(range(len(P)), 2):
        X = triangulate_nviews(P[[cam_i, cam_j]], pts_2d[[cam_i, cam_j]])
        errors = triangulate_n_views_error(P, pts_2d, X)
        inliners = np.nonzero(errors < inliner_thr)[0]
        if tuple(inliners.tolist()) in checked_inliner_set:
            continue
        
        checked_inliner_set.add(tuple(inliners.tolist()))
        
        if len(inliners) == 0:
            continue
        
        X_in = triangulate_nviews(P[inliners], pts_2d[inliners])
        errors_in = triangulate_n_views_error(P[inliners], pts_2d[inliners], X_in)
        
        if errors_in.mean() < best_fit_error:
            best_inliners = inliners
            best_X = X_in
            
    # if best_inliners is not None and tuple(best_inliners) != tuple(np.arange(len(P))):
    #     print(len(P), best_inliners)
            
    # we need to return a value even when all combinations fail
    if best_inliners is None:
        best_X = triangulate_nviews(P, pts_2d)
        best_inliners = np.arange(len(P))
           
    errors = triangulate_n_views_error(P, pts_2d, best_X)
        
    cam_used = np.zeros(len(P), dtype=bool)
    cam_used[best_inliners] = True
    
    return best_X, cam_used, errors




def parse_patch(patch_def:List[str]):
    ''' parse patch file and create a list of markers
        
        # return: 
            marker_def, patch
        
        **marker_def**: a list of marker definitions, each marker def is a list containing one or two tuples, 
        the first element of the tuple is a string indicating the name that contains the marker
        the second element of the tuple is an integer in [0,4) indicating the index of the marker in the block 
                
            [ 
                \# marker_def 0:
                ( (block name 1, index of this marker in the block 1) ),
                \# marker_def 1:
                ( (block name 1, index of this marker in the block 1),
                (block name 2, index of this marker in the block 2)  ),
                \# marker_def ...
            ]
            
        in  current settings, a marker is either contained by a single block, or shared by two blocks,
        so each marker_def contains either one or two block tuples
                    
        **patch**: a 2D list, the array of block names of a patch. 
    '''
    
    if len(patch_def) == 0:
        return [], []
    
    patch = [textwrap.wrap(row, width=2) for row in patch_def]
    n_rows = len(patch)
    n_cols = len(patch[0])
    
    assert all(len(row) == n_cols for row in patch[1:])
    
    marker_def = [[[] for c in range(n_cols+1)] for r in range(n_rows+1)]
    
    ################
    #  marker index:
    #  0._______.1
    #   |       |
    #   |       |
    #   |_______|
    #  3         2
    ################
    for r, row in enumerate(patch):
        for c, blk in enumerate(row):
            if blk == '**':
                continue
            marker_def[r  ][c  ].append((blk, 0))
            marker_def[r  ][c+1].append((blk, 1))
            marker_def[r+1][c+1].append((blk, 2))
            marker_def[r+1][c  ].append((blk, 3))
            
    marker_def = sum(marker_def, [])
    marker_def = [tuple(marker) for marker in marker_def if len(marker) > 0]
    
    return marker_def, patch
        
    
def load_label_patches(fn:str):
    '''
    return: marker_defs, blocks, patches
    
        **marker_def**: a list of marker definitions, each marker def is a list containing one or two tuples, 
        the first element of the tuple is a string indicating the name that contains the marker
        the second element of the tuple is an integer in [0,4) indicating the index of the marker in the block 
                
            [ 
                \# marker_def 0:
                ( (block name 1, index of this marker in the block 1) ),
                \# marker_def 1:
                ( (block name 1, index of this marker in the block 1),
                (block name 2, index of this marker in the block 2)  ),
                \# marker_def ...
            ]
            
        in  current settings, a marker is either contained by a single block, or shared by two blocks,
        so each marker_def contains either one or two block tuples
        
        **blocks**: a dict: 
            {
                "blk_name": ( [i,j,k,l], patch_name ),
            }
        where [i,j,k,l] is the index of markers of the block, in the order like                     
                ################
                #  marker index:
                #  0._______.1
                #   |       |
                #   |       |
                #   |_______|
                #  3         2
                ################
    
        **patches**: a dict of block arrays, key is the name of the patch
    '''
    
    with open(fn) as f:
        patches_def:dict = json.load(f)
        
    print(list(patches_def.keys()))
    
    # process patches
    marker_defs = [] # marker_def: (blk_name, idx_in_blk)
    patches = {}
    for name, patch_def in patches_def.items():
        markers, patch = parse_patch(patch_def)
        marker_defs.extend(markers)
        patches[name] = patch
        
    block_names = [(blk_name, patch_name) for patch_name, patch in patches.items() 
                   for blk_name in itertools.chain.from_iterable(patch) 
                   if blk_name != '**']
    
    blocks = {blk_name: {'markers': [0,0,0,0], 'patch': patch_name} for blk_name, patch_name in block_names}
    for idx, marker_info in enumerate(marker_defs):
        for marker in marker_info:
            blk_name, idx_in_blk = marker
            blocks[blk_name]['markers'][idx_in_blk] = idx
        
    return marker_defs, blocks, patches
    
def get_edges_from_block_def(block_defs:Dict[str, Dict[str, List|str]]):
    # 
    edges = set()
    for blk_name, blk_def in block_defs.items():
        indices = blk_def['markers']
        for i, j in zip(indices[:-1], indices[1:]):
            edges.add((i,j))
        edges.add((indices[-1], indices[0]))
        
    edges = list(edges)
    return edges
    

def find_index_in_nested_list(nested_list:List[Any|List[Any]], item:Any):
    def recursive_search(lst, item, path=()):
        for index, element in enumerate(lst):
            current_path = path + (index,)
            if element == item:
                return current_path
            elif isinstance(element, list):
                result = recursive_search(element, item, current_path)
                if result:
                    return result
        return None
    return recursive_search(nested_list, item)

def draw_block_code(image:np.ndarray, markers:np.ndarray, blocks:np.ndarray, label_0_char, label_1_char, blk_dir, wrap_text:bool, text_color:float=1.0):
    for blk_idx, blk in enumerate(blocks):
        corners = markers[blk,:]
        label = label_0_char[blk_idx][0] + label_1_char[blk_idx][0]
        label_confidence = label_0_char[blk_idx][1] * label_1_char[blk_idx][1]
        
        if label_confidence < 0.5:
            continue
        
        direction = blk_dir[blk_idx][0]
        dir_confidence = blk_dir[blk_idx][1]
        
        pt = np.round(np.mean(corners, axis=0)).astype(int)
        
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
            c = text_color
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
            
            c = label_confidence * text_color
            text_img = np.zeros((text_h+1, text_w+1), dtype=image.dtype)
            cv2.putText(text_img, label_text, (0, text_h), cv2.FONT_HERSHEY_PLAIN, h, c, 1, cv2.LINE_AA)
            
            img_blk = image[max(0, pt[1]-text_img.shape[0]//2):pt[1]+text_img.shape[0]//2,
                            max(0, pt[0]-text_img.shape[1]//2):pt[0]+text_img.shape[1]//2]
            text_img = text_img[:img_blk.shape[0], :img_blk.shape[1]]
            text_mask = text_img > 0
            
            img_blk[text_mask] = text_img[text_mask]
            img_blk[:] = text_img
        
    return image


class VideoWriter:
    def __init__(self, out_filename, width, height, fps, pix_fmt='rgb24', verbose=False) -> None:
        self.process = self.start_ffmpeg_process(out_filename, width, height, fps, pix_fmt, verbose)        
    
    @staticmethod
    def get_video_size(filename):
        probe = ffmpeg.probe(filename)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        
        # import json
        # print(json.dumps(video_info, indent=4))
        return width, height
    
    @staticmethod
    def get_num_frames(filename):
        probe = ffmpeg.probe(filename)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        num_frames = int(video_info['nb_frames'])
        return num_frames
    
    
    def start_ffmpeg_process(self, out_filename, width, height, fps, pix_fmt='rgb24', verbose=False):
        video = ffmpeg.input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s='{}x{}'.format(width, height), framerate=1)
        video = video.filter('fps', fps=fps, round='up')
        video = video.setpts(f'1/{fps}*PTS')
        if verbose:
            video = video.output(out_filename, pix_fmt='yuv420p')    
        else:
            video = video.output(out_filename, pix_fmt='yuv420p', loglevel='quiet')
        video = video.overwrite_output()
        
        args = video.compile()
        if verbose:
            print(args)
        return subprocess.Popen(args, stdin=subprocess.PIPE)
    
    def write_frame(self, frame:np.array):
        # frame: (height, width, 3)
        frame = np.ascontiguousarray(frame)
        self.process.stdin.write(frame.data)
        
    def write(self, frame:np.ndarray):
        self.write_frame(frame)
    
    def release(self):
        self.process.stdin.close()
        self.process.wait()
        
        
class JSONEncoderWithNumpy(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONEncoderWithNumpy, self).default(obj)
        
        
if __name__ == '__main__':
    # __test()
    find_quads(np.random.uniform(size=(60, 2))*(2048,2048), distance_thr=0.2*2048, min_area=0.1*0.1*2048*2048, max_area=0.2*0.2*2048*2048)