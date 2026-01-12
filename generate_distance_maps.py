import os
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt




"""
2D距离图生成与保存工具
功能：批量处理二值掩码图像，生成对应的距离图并保存为.npy文件
"""

import os
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image
import concurrent.futures
from tqdm import tqdm
import argparse


def generate_single_distance_map(mask_path, output_dir, normalize=True, check_blank=True):
    """
    处理单个掩码文件生成距离图

    参数：
        mask_path: str - 掩码文件路径
        output_dir: str - 输出目录
        normalize: bool - 是否归一化到[0,1]
        check_blank: bool - 是否跳过全黑/全白掩码

    返回：
        success: bool - 是否成功处理
    """
    try:
        # 1. 加载掩码
        mask = np.array(Image.open(mask_path).convert('L'))

        # 2. 检查无效掩码
        if check_blank:
            unique_vals = np.unique(mask)
            if len(unique_vals) == 1:  # 全黑或全白
                return False

        # 3. 二值化处理（兼容不同格式的掩码）
        binary_mask = (mask > 127).astype(np.uint8)

        # 4. 计算距离图
        fg_dist = distance_transform_edt(binary_mask)
        bg_dist = distance_transform_edt(1 - binary_mask)
        # dist_map = np.minimum(fg_dist, bg_dist)
        dist_map = np.where(binary_mask == 1, fg_dist, bg_dist)

        # 5. 归一化处理
        if normalize:
            max_val = np.max(dist_map)
            if max_val > 0:
                dist_map = dist_map / max_val

        # 6. 保存为.npy文件
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(mask_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(output_path, dist_map.astype(np.float32))

        return True

    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")
        return False


def batch_generate_distance_maps(mask_dir, output_dir, num_workers=8, **kwargs):
    """
    批量生成距离图

    参数：
        mask_dir: str - 掩码目录路径
        output_dir: str - 输出目录路径
        num_workers: int - 并行工作进程数
        **kwargs: 传递给generate_single_distance_map的参数
    """
    # 获取所有掩码文件
    supported_ext = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
    mask_paths = [
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
        if os.path.splitext(f)[1].lower() in supported_ext
    ]

    print(f"发现 {len(mask_paths)} 个掩码文件需要处理")

    # 使用多线程加速
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for mask_path in mask_paths:
            futures.append(executor.submit(
                generate_single_distance_map,
                mask_path=mask_path,
                output_dir=output_dir,
                **kwargs
            ))

        # 进度条显示
        success_count = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            success_count += future.result()

    print(f"成功处理 {success_count}/{len(mask_paths)} 个文件")


def verify_distance_map(mask_path, dist_path):
    """验证距离图生成是否正确"""
    import matplotlib.pyplot as plt

    # 加载原始数据
    mask = np.array(Image.open(mask_path).convert('L'))
    dist_map = np.load(dist_path)

    # 可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(mask, cmap='gray')
    plt.title("Original Mask")

    plt.subplot(132)
    plt.imshow(dist_map, cmap='jet')
    plt.colorbar()
    plt.title("Distance Map")

    plt.subplot(133)
    plt.imshow(mask, cmap='gray', alpha=0.5)
    plt.imshow(dist_map, cmap='jet', alpha=0.5)
    plt.title("Overlay")

    plt.tight_layout()
    plt.show()

    print(f"距离图统计 - Min: {dist_map.min():.4f}, Max: {dist_map.max():.4f}, Mean: {dist_map.mean():.4f}")


if __name__ == "__main__":
    # 命令行参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask_dir", type=str, default="/mnt/d/2023/zjy/VMB-UNet/data/isic2017/train/masks")
    parser.add_argument("--output_dir", type=str, default="/mnt/d/2023/zjy/VMB-UNet/data/isic2017/train/distance_maps")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--verify", type=str, default=None)
    args = parser.parse_args(args=[])
    os.makedirs(args.output_dir, exist_ok=True)


    if args.verify:
        # 验证模式
        base_name = os.path.splitext(os.path.basename(args.verify))[0]
        dist_path = os.path.join(args.output_dir, f"{base_name}.npy")
        verify_distance_map(args.verify, dist_path)
    else:
        # 批量生成模式
        batch_generate_distance_maps(
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            normalize=True,
            check_blank=True
        )