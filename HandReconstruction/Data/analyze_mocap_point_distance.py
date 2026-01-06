"""
点云距离分析工具

此脚本用于分析mocap数据中每一帧点云的空间分布特性。主要功能包括：
1. 加载并处理mocap点云数据，过滤掉无效点(-1000)
2. 计算每一帧中点对之间的最大距离和平均距离
3. 计算每一帧中可见点的比例
4. 绘制并保存最大距离、平均距离和可见点比例随时间变化的图表
5. 输出距离的统计信息(均值、标准差、最大值、最小值)

输入：
- 通过config.py指定的点云数据路径和参数

输出 (保存在 Statistics/ 目录下)：
- max_point_distances.png: 每帧点云中最远两点距离的变化图
- avg_point_distances.png: 每帧点云中所有点对平均距离的变化图
- visibility_ratio.png: 每帧可见点占总点数的比例变化图
- 控制台输出的统计信息和异常帧的点对索引

使用方法：
直接运行此脚本: python Data/analyze_mocap_point_distance.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import config

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像负号'-'显示为方块的问题


def analyze_point_distances():
    """
    Analyzes point cloud distances and visibility over frames.

    Loads mocap point cloud data, calculates max/avg distances and visibility ratio
    for each frame, plots these metrics, and saves the plots to the 'Statistics/' directory.

    Returns:
        tuple:
        - max_distances (list): List of maximum point distances for each frame.
        - avg_distances (list): List of average point distances for each frame.
        - visibility_ratios (list): List of visibility ratios for each frame.
        - max_point_pairs (list): List of tuples containing indices of the points with max distance for each frame.
    """
    # 创建输出目录
    output_dir = "Statistics"
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("加载数据...")
    with open(config.data["hand_mocap_point_data_path"], "rb") as f:
        points = np.load(f)  # [num_frames, num_points, 3]
        
    base_name = os.path.basename(config.data["hand_mocap_point_data_path"])
    base_name = os.path.splitext(base_name)[0]

    # 应用裁剪参数
    cutoff_start = config.data["cutoff_start"]
    cutoff_end = config.data["cutoff_end"]
    cutoff_step = config.data["cutoff_step"]
    points = points[cutoff_start:cutoff_end:cutoff_step]

    frame_count = points.shape[0]
    marker_count_total = points.shape[1] # Renamed for clarity
    print(f"数据: 总帧数: {frame_count}, 每帧总标记点数: {marker_count_total}")

    # 存储每帧的指标
    max_distances = []
    avg_distances = []
    visibility_ratios = []
    max_point_pairs = []  # 存储具有最大距离的点对索引

    # 无效点值
    invalid_point_value = config.data["invalid_point_value"]

    # 处理每一帧
    for i in tqdm(range(frame_count), desc="计算指标"):
        frame_points = points[i]
        num_total_points = frame_points.shape[0]

        # 过滤掉无效点
        valid_mask = ~np.all(np.isclose(frame_points, invalid_point_value), axis=1)
        valid_points = frame_points[valid_mask]
        num_valid_points = len(valid_points)

        # 计算可见点比例
        visibility_ratio = num_valid_points / num_total_points if num_total_points > 0 else 0
        visibility_ratios.append(visibility_ratio)

        if num_valid_points <= 1:
            # 如果只有0或1个有效点，则无法计算距离
            max_distances.append(0)
            avg_distances.append(0)
            max_point_pairs.append((-1, -1))
            continue

        # 计算点之间的所有成对距离
        distances = pdist(valid_points)

        # 获取最大距离和平均距离
        max_dist = np.max(distances)
        avg_dist = np.mean(distances)
        max_distances.append(max_dist)
        avg_distances.append(avg_dist)

        # 找到具有最大距离的点对
        dist_matrix = squareform(distances)
        max_indices = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        # 将局部索引转换回原始点索引
        valid_indices = np.where(valid_mask)[0]
        max_point_pairs.append(
            (valid_indices[max_indices[0]], valid_indices[max_indices[1]])
        )

    # 绘制最大距离图
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(frame_count), max_distances)
    plt.xlabel("帧数")
    plt.ylabel("最大点距离 (单位)")
    plt.title("每帧中最远两点的距离")
    plt.grid(True)

    # 计算最大距离的统计数据
    mean_max_dist = np.mean(max_distances)
    std_max_dist = np.std(max_distances)
    min_max_dist = np.min(max_distances)
    max_max_dist = np.max(max_distances)

    stats_text = f"统计数据:\n均值: {mean_max_dist:.2f}\n标准差: {std_max_dist:.2f}\n最小值: {min_max_dist:.2f}\n最大值: {max_max_dist:.2f}"
    plt.figtext(
        0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_max_point_distances.png"), dpi=300)
    plt.close()

    # 绘制平均距离图
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(frame_count), avg_distances)
    plt.xlabel("帧数")
    plt.ylabel("平均点距离 (单位)")
    plt.title("每帧中所有点对的平均距离")
    plt.grid(True)

    # 计算平均距离的统计数据
    mean_avg_dist = np.mean(avg_distances)
    std_avg_dist = np.std(avg_distances)
    min_avg_dist = np.min(avg_distances)
    max_avg_dist = np.max(avg_distances)

    stats_text = f"统计数据:\n均值: {mean_avg_dist:.2f}\n标准差: {std_avg_dist:.2f}\n最小值: {min_avg_dist:.2f}\n最大值: {max_avg_dist:.2f}"
    plt.figtext(
        0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_avg_point_distances.png"), dpi=300)
    plt.close()

    # 绘制可见点比例图
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(frame_count), visibility_ratios)
    plt.xlabel("帧数")
    plt.ylabel("可见点比例 (可见点数 / 总点数)")
    plt.title("每帧可见标记点比例")
    plt.ylim(0, 1.05) # Set y-axis limit from 0 to 1
    plt.grid(True)

    # 计算可见点比例的统计数据
    mean_vis_ratio = np.mean(visibility_ratios)
    std_vis_ratio = np.std(visibility_ratios)
    min_vis_ratio = np.min(visibility_ratios)
    max_vis_ratio = np.max(visibility_ratios)

    stats_text_vis = f"统计数据:\n均值: {mean_vis_ratio:.2%}\n标准差: {std_vis_ratio:.2%}\n最小值: {min_vis_ratio:.2%}\n最大值: {max_vis_ratio:.2%}"
    plt.figtext(
        0.02, 0.02, stats_text_vis, fontsize=10, bbox=dict(facecolor="white", alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_visibility_ratio.png"), dpi=300)
    plt.close()

    print(
        f"最大距离统计: 均值={mean_max_dist:.2f}, 标准差={std_max_dist:.2f}, 最小值={min_max_dist:.2f}, 最大值={max_max_dist:.2f}"
    )
    print(
        f"平均距离统计: 均值={mean_avg_dist:.2f}, 标准差={std_avg_dist:.2f}, 最小值={min_avg_dist:.2f}, 最大值={max_avg_dist:.2f}"
    )
    print(
        f"可见点比例统计: 均值={mean_vis_ratio:.2%}, 标准差={std_vis_ratio:.2%}, 最小值={min_vis_ratio:.2%}, 最大值={max_vis_ratio:.2%}"
    )

    return max_distances, avg_distances, visibility_ratios, max_point_pairs


if __name__ == "__main__":
    # 如果中文显示有问题，尝试打印可用字体
    try:
        max_distances, avg_distances, visibility_ratios, max_point_pairs = analyze_point_distances()
    except Exception as e:
        print(f"出现错误: {e}")
        print("\n尝试检查可用字体...")
        from matplotlib.font_manager import fontManager

        fonts = [f.name for f in fontManager.ttflist]
        print("系统可用字体:")
        for font in sorted(fonts):
            print(font)

        # 尝试使用另一种中文字体
        print("\n尝试使用替代字体...")
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimSun",
            "FangSong",
            "KaiTi",
            "STHeiti",
        ]
        max_distances, avg_distances, visibility_ratios, max_point_pairs = analyze_point_distances()

    # 输出几个最大值和最小值的帧索引，便于调试
    max_distance_frames = np.argsort(max_distances)[-5:][::-1]
    min_distance_frames = np.argsort(max_distances)[:5]

    print("\n最大距离的帧索引:")
    for idx in max_distance_frames:
        print(
            f"帧 {idx}: 距离 = {max_distances[idx]:.2f}, 点对 = {max_point_pairs[idx]}"
        )

    print("\n最小距离的帧索引:")
    for idx in min_distance_frames:
        print(
            f"帧 {idx}: 距离 = {max_distances[idx]:.2f}, 点对 = {max_point_pairs[idx]}"
        )

    # 输出可见点比例较低的帧索引 (Optional: Helps identify frames with potential tracking issues)
    min_visibility_frames = np.argsort(visibility_ratios)[:5]
    print("\n可见点比例最低的帧索引:")
    for idx in min_visibility_frames:
        print(
            f"帧 {idx}: 比例 = {visibility_ratios[idx]:.2%}"
        )
