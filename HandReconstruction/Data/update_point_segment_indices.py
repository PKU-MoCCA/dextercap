import numpy as np

indices = np.load("point_segment_idx-2025.01.npy")
indices[indices == -1] = 16  # Temporarily change -1 to 16 to avoid conflict
np.save("point_segment_idx-2025.01.npy", indices)
print("Successfully updated point_segment_idx.npy")
