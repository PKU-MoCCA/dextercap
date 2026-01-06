import numpy as np
import json
import matplotlib.pyplot as plt
from utils import load_label_patches

def get_point_idx(patches_file, pts_file):
    marker_defs, block_defs, patches = load_label_patches(patches_file)

    # print(len(marker_defs), marker_defs, "\n")
    # print(len(block_defs), block_defs, "\n")
    # print(len(patches), patches, "\n")

    parts = [9,8,7, 12,11,10, 6,5,4, 3,2,1, 15,14,13, 13,0,0,16, 20,21,22,23,24,25]  # need to be changed
    part2idx = {}
    for i, (name, _) in enumerate(patches.items()):
        part2idx[name] = parts[i]

    num_point = len(marker_defs)
    num_block = len(block_defs)
    idx = [-1 for _ in range(num_point)]

    for blk_name, blk_def in block_defs.items():
        for marker_idx in blk_def['markers']:
            if idx[marker_idx] == -1:
                idx[marker_idx] = part2idx[blk_def['patch']]
            else:
                assert idx[marker_idx] == part2idx[blk_def['patch']]

    idx = np.array(idx)
    np.save("result/0428/idx.npy", idx)


def remove_outliers_patch(data0, idx_path, max_len=0.01):
    data = data0.copy()
    indices = np.load(idx_path)
    frame_num, point_num, _ = data.shape
    unique_classes = np.unique(indices)

    remove_points = 0

    for frame in range(frame_num):
        if frame % 500 == 0:
            print(f"Patch processing frame {frame} / {frame_num}")

        for idx in unique_classes:
            class_indices = np.where((indices == idx))[0]
            class_points = data[frame, class_indices, :]
            
            valid_mask = (class_points != [-1000, -1000, -1000]).all(axis=1)
            valid_points = class_points[valid_mask]
            # print(valid_points.shape)
            valid_indices = class_indices[valid_mask]
            k = valid_points.shape[0]  # number of valid points
            
            if k <= 1:
                data[frame, valid_indices] = [-1000, -1000, -1000]
                continue
            
            # distance matrix
            diff = valid_points[:, np.newaxis, :] - valid_points[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))    # [m, m]
            
            # adjacency matrix
            adj = (distances <= max_len)
            np.fill_diagonal(adj, False)
            # print(adj)
                        
            rows, cols = np.triu_indices(k, 1)  # indices of right-up-triangluar of a k*k matrix
            edge_mask = adj[rows, cols]
            # print(rows)
            # print(cols)
            # print(edge_mask)
            # print(rows[edge_mask])
            # print(cols[edge_mask])
            
            edges = np.column_stack((rows[edge_mask], cols[edge_mask]))
            # print(edges.shape)

            # union-find set
            parent = list(range(k))
            
            def find(u):
                while parent[u] != u:
                    parent[u] = parent[parent[u]] # path compression
                    u = parent[u]
                return u
            
            def union(u, v):
                root_u = find(u)
                root_v = find(v)
                if root_u != root_v:
                    parent[root_v] = root_u

            for u, v in edges:
                union(u, v)
            
            roots = [find(i) for i in range(k)]
            unique_roots, counts = np.unique(roots, return_counts=True)

            if len(unique_roots) == 0:
                print("why???")
                print(roots, unique_roots, counts)
                continue

            max_root = unique_roots[np.argmax(counts)]
            
            for i in range(k):
                if find(i) != max_root:
                    remove_points += 1
                    original_idx = valid_indices[i]
                    data[frame, original_idx] = [-1000, -1000, -1000]
    
    print(f"Removed points: {remove_points} / {frame_num} = {remove_points / frame_num}")
    return data


def remove_outliers_window(data, discontinuity=[], window_size=30, z_threshold=2):
    """
    check and remove outliers, set to [-1000, -1000, -1000]

    Args:
        data: ndarray, shape should be [frame_num, point_num, 3]
        discontinuity: list of discontinuity frame
        window_size (int)
        z_threshold (float)

    Returns:
        result: ndarray, shape should be [frame_num, point_num, 3]
    """
    frame_num, point_num, _ = data.shape
    result = data.copy()
    remove_points = 0

    discontinuity = [0] + discontinuity + [frame_num]
    print(discontinuity)

    for i in range(len(discontinuity) - 1):
        for frame in range(discontinuity[i], discontinuity[i+1]):
            if frame % 500 == 0:
                print(f"Windows processing frame {frame} / {frame_num}")

            start = max(discontinuity[i], frame - window_size // 2)
            end = min(discontinuity[i+1], frame + window_size // 2 + 1)
            window_data = data[start:end]

            for point in range(point_num):
                if np.all(data[frame, point] == [-1000, -1000, -1000]):
                    continue    

                point_coords = window_data[:, point, :]
                valid_coords = point_coords[~np.all(point_coords == [-1000, -1000, -1000], axis=1)]

                # if len(valid_coords) <= 1:  # Only this frame's point can be seen, it's weird! Remove it!
                #     result[frame, point] = [-1000, -1000, -1000]
                #     remove_points += 1
                # else:
                mean = np.mean(valid_coords, axis=0)
                std = np.std(valid_coords, axis=0)
                current_point = data[frame, point]

                z_score = np.abs((current_point - mean) / (std + 1e-6))
                if np.any(z_score > z_threshold):
                    result[frame, point] = [-1000, -1000, -1000]
                    remove_points += 1

    print(f"remove points: {remove_points} / {frame_num} = {remove_points/frame_num}")
    return result


def remove_outliers_object(data0, max_len=0.087):
    data = data0.copy()
    frame_num, point_num, _ = data.shape
    remove_points = 0

    for frame in range(frame_num):
        if frame % 500 == 0:
            print(f"Patch processing frame {frame} / {frame_num}")

        frame_points = data[frame, :, :]        
        frame_indices = np.arange(point_num)
        valid_mask = (frame_points != [-1000, -1000, -1000]).all(axis=1)
        valid_points = frame_points[valid_mask]
        valid_indices = frame_indices[valid_mask]
        k = valid_points.shape[0]  # number of valid points
        
        if k <= 1:
            continue
        
        # distance matrix
        diff = valid_points[:, np.newaxis, :] - valid_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))    # [m, m]
        
        # adjacency matrix
        adj = (distances <= max_len)
        np.fill_diagonal(adj, False)
                    
        rows, cols = np.triu_indices(k, 1)  # indices of right-up-triangluar of a k*k matrix
        edge_mask = adj[rows, cols]
        
        edges = np.column_stack((rows[edge_mask], cols[edge_mask]))

        # union-find set
        parent = list(range(k))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]] # path compression
                u = parent[u]
            return u
        
        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u

        for u, v in edges:
            union(u, v)
        
        roots = [find(i) for i in range(k)]
        unique_roots, counts = np.unique(roots, return_counts=True)

        max_root = unique_roots[np.argmax(counts)]
        
        for i in range(k):
            if find(i) != max_root:
                remove_points += 1
                original_idx = valid_indices[i]
                data[frame, original_idx] = [-1000, -1000, -1000]

    print(f"Removed points: {remove_points} / {frame_num} = {remove_points / frame_num}")
    return data


def interpolate_points(data, discontinuity=[], invalid_value=[-1000, -1000, -1000]):
    """
    Args:
        data (np.ndarray): [frame_num, point_num, 3] 
        invalid_value (list): [-1000, -1000, -1000]

    Returns:
        np.ndarray: [frame_num, point_num, 3]
    """
    frame_num, point_num, _ = data.shape
    result = data.copy()
    interpolate_points = 0
    discontinuity = [0] + discontinuity + [frame_num]
    print(discontinuity)

    for point in range(point_num):
        if point % 500 == 0:
            print(f"Interpolating point {point} / {point_num}")

        for i in range(len(discontinuity) - 1):
            start, end = discontinuity[i], discontinuity[i+1]
            for frame in range(start, end):
                if np.all(data[frame, point] == invalid_value):
                    front_weight = 0
                    for offset in [-1, -2]:
                        front_frame = frame + offset
                        if start <= front_frame < end and not np.all(data[front_frame, point] == invalid_value):
                            front_weight = 3 + offset
                            break

                    back_weight = 0
                    for offset in [1, 2]:
                        back_frame = frame + offset
                        if start <= back_frame < end and not np.all(data[back_frame, point] == invalid_value):
                            back_weight = 3 - offset
                            break

                    if front_weight and back_weight:
                        front = np.array(data[front_frame, point])
                        back  = np.array(data[back_frame, point])
                        result[frame, point] = (front * front_weight + back * back_weight) / (front_weight + back_weight)
                        interpolate_points += 1

    print(f"interpolate points: {interpolate_points} / {frame_num} = {interpolate_points / frame_num}")
    return result


def remove_outliers(path, idx_path, date):
    data = np.load(path)
    discontinuity = []
    observed_points_per_frame(data)
    r1 = remove_outliers_patch(data, idx_path=idx_path, max_len=0.01)
    r3 = remove_outliers_window(r1, discontinuity)
    # r3 = remove_outliers_object(r2)
    r4 = interpolate_points(r3, discontinuity)
    observed_points_per_frame(r4)
    np.save(f"result/{date}/pts_final.npy", r4)
    
    idx = np.load(idx_path)
    print(idx.shape, r3.shape)
    mask = idx < 20
    idx_hand, idx_obj = idx[idx<20], idx[idx>=20]
    pts_hand = np.array([[r3[i][j] for j in range(len(idx)) if idx[j] < 20] for i in range(len(r3))])
    pts_obj = np.array([[r3[i][j] for j in range(len(idx)) if idx[j] >= 20] for i in range(len(r3))])
    
    print(pts_hand.shape, pts_obj.shape)

    np.save(f"result/{date}/pts_hand.npy", pts_hand)
    np.save(f"result/{date}/idx_hand.npy", idx_hand)
    np.save(f"object/{date}/pts_obj.npy", pts_obj)
    np.save(f"object/{date}/idx_obj.npy", idx_obj)



def statistic_edge_length(videos=[1,2,3,4,5,6,7,8,9,10,11,13,14]):
    minE, maxE = 1000, -1
    length = []
    for v in videos:
        json_path = f"dataset/mocap0428/jsons/{v}.json"
        print(json_path)
        with open(json_path, "r") as f:
            data = json.load(f)
        for frame in data:
            print(frame["blocks"].__len__(), frame["keypoints"].__len__())
            for b in frame["blocks"]:
                pts = np.array(b["corners"])
                for i in range(4):
                    len = np.linalg.norm(pts[i] - pts[(i+1)%4])
                    length.append(len)
                    if len > maxE: maxE = len
                    if len < minE: minE = len
    print(minE, maxE, length.__len__())
    plt.hist(length, bins=20, edgecolor='black', alpha=0.7)
    plt.savefig("statistic.png")


def observed_points_per_frame(pts):
    observed = ~np.all(pts == -1000, axis=2)
    observed_points_per_frame = np.sum(observed, axis=1)
    average_observed_points = np.mean(observed_points_per_frame)
    print(average_observed_points)


def statistic_point_seen(pts_path, patch_path):
    marker_defs, block_defs, patches = load_label_patches(patch_path)
    idx = np.load(pts_path)
    for i in range(len(idx)):
        for j in range(len(idx[i])):
            if idx[i][j] < 20:
                print(idx[i][j])    


if __name__ == "__main__":
    remove_outliers("result/0428/pts.npy", "result/0428/idx.npy", "0428")
    # statistic_edge_length()
    # get_point_idx("dataset/mocap0428/patches_0428.json", "result/0428/pts.npy")

