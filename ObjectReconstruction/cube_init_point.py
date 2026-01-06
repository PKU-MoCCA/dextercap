import json

import numpy as np

size = 0.005
length, w, h = 0.15, 0.04, 0.04

# The initial positions of the upper-left corners of each cube face, adjusted by offsets
left_up = {
    "cube_up": np.array([-length / 2, h / 2, w / 2]) + np.array([0.057, 0, -0.012]),
    "cube_down": np.array([-length / 2, -h / 2, w / 2]) + np.array([0.01, 0, -0.0125]),
    "cube_left": np.array([-length / 2, h / 2, -w / 2]) + np.array([0, -0.013, 0.009]),
    "cube_right": np.array([length / 2, h / 2, -w / 2]) + np.array([0, -0.008, 0.007]),
    "cube_front": np.array([length / 2, h / 2, w / 2]) + np.array([-0.062, -0.0108, 0]),
    "cube_back": np.array([length / 2, h / 2, -w / 2]) + np.array([-0.011, -0.01, 0]),
}

# The step vectors for each face: [right direction vector, down direction vector] * size
# Each entry is a (2, 3) ndarray
# Used to compute the grid of points on each face

delta = {  # right  down
    "cube_up": np.array([[-1, 0, 0], [0, 0, -1]]) * size,
    "cube_down": np.array([[1, 0, 0], [0, 0, -1]]) * size,
    "cube_left": np.array([[0, 0, 1], [0, -1, 0]]) * size,
    "cube_right": np.array([[0, -1, 0], [0, 0, 1]]) * size,
    "cube_front": np.array([[1, 0, 0], [0, -1, 0]]) * size,
    "cube_back": np.array([[-1, 0, 0], [0, -1, 0]]) * size,
}

# print(left_up)


def main():
    """
    Generate and save the initial 3D positions of points on the cube's surface based on patch masks.

    Reads a JSON file containing patch masks for each cube face, computes the 3D positions of valid points
    according to the mask and face geometry, and saves the result as a numpy array.
    """
    with open("dataset/mocap0414/patches_0414.json") as f:
        patches = json.load(f)

    points_positions = []  # List to store valid 3D point positions

    for i in patches:
        if not i.startswith("cube"):
            continue  # Skip non-cube entries
        # print(patches[i])
        rows = len(patches[i]) + 1  # Number of rows in the grid (patch rows + 1)
        cols = (
            len(patches[i][0]) // 2 + 1
        )  # Number of columns in the grid (patch cols/2 + 1)

        # position: (rows, cols, 3) array, each entry is a 3D coordinate
        position = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                # Compute the 3D position for each grid point
                position[r][c] = left_up[i] + (delta[i][0] * c + delta[i][1] * r)

        # mask: (rows, cols) array, 1 if the point is valid, 0 otherwise
        mask = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows - 1):
            for c in range(cols - 1):
                # If the patch is not masked (not '**'), mark the four corners as valid
                if patches[i][r][c * 2 : c * 2 + 2] != "**":
                    mask[r][c] = 1
                    mask[r + 1][c] = 1
                    mask[r][c + 1] = 1
                    mask[r + 1][c + 1] = 1

        # print(i, mask)

        for r in range(rows):
            for c in range(cols):
                if mask[r][c] == 1:
                    points_positions.append(position[r][c])

    points_positions = np.array(points_positions)  # (N, 3) array of valid 3D points
    print(points_positions.shape)
    # print(points_positions)
    np.save("object/0414/cube_init.npy", points_positions)


if __name__ == "__main__":
    main()
