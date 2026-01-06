import numpy as np
import json
import os
import cv2


def extract_frame(video_path, frame_index, img_path):
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cv2.imwrite(img_path, frame)
    cap.release()


def modify_json(json_path, img_path, frame_index):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    for item in data:
        item["image"]["path"] = img_path
        item["frame"] = frame_index


def save_frames(total_path, origin_folder, img_folder, new_json_folder, video_idx):
    for v in video_idx:
        print(f"Processing video {v}")

        video_path = os.path.join(total_path, origin_folder, f"{v}.avi")
        json_path = os.path.join(total_path, origin_folder, f"{v}.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        
        for item in data:
            frame_index = item["frame"]
            print(f"\tframe {frame_index}")
            img_name = f"{int(v)}_{int(frame_index)}.png"
            img_path = os.path.join(img_folder, img_name)    
            extract_frame(video_path, frame_index, os.path.join(total_path, img_path))
            item["image"] = {}
            item["image"]["path"] = img_path
            item["image"]["size"] = [2048, 2448]

        new_json_path = os.path.join(total_path, new_json_folder, f"{v}.json")
        with open(new_json_path, "w") as f:
            json.dump(data, f, indent=4)


def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def get_marker_indices(patch_file):
    with open(patch_file, "r") as f:
        data = json.load(f)
    from utils import parse_patch
    marker_defs, patches = parse_patch(data)
    print(marker_defs)
    print(patches)
    return marker_defs

if __name__ == "__main__":
    save_frames(
        "dataset/mocap0414",
        "origin", "imgs", "jsons", 
        [1,2,3,4,5,6,7,8,9,10,11,13,14]
    )
