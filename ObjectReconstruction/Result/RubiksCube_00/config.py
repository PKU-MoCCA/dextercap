# The unique string represents this mocap session
mocap_session_name = "RubiksCube_00"

# Object data 7D, including x, y, z, quaternion in xyzw
data = {
    # Invalid coordinates. If the object's coordinates are invalid, visualization will not be performed.
    "invalid_point_value": [-1000, -1000, -1000],
    # Object size, unit: m
    "object_size": [0.051, 0.051, 0.051],
    # The frame range of the object, left closed and right open. If the object exists in all frames, then object_frame_range = [0, -1]
    "object_frame_range": [0, -1],
}
