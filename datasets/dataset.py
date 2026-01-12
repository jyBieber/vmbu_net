from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random
import torch
from scipy import ndimage
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
def debug_visualize(img_tensor, msk_tensor, dist_tensor, title='Debug Visualization'):
    # 将图像转换为 HWC 格式，并取 CPU 上的 numpy 数组
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    msk = msk_tensor.squeeze().cpu().numpy()
    dist = dist_tensor.cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("增强后的图像")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(msk, cmap='gray')
    plt.title("增强后的掩码")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(msk, cmap='gray', alpha=0.5)
    plt.imshow(dist, cmap='jet', alpha=0.5)
    plt.title("掩码与距离图叠加")
    plt.axis("off")

    plt.suptitle(title)
    plt.show()

    print(f"距离图统计信息 - Min: {dist.min():.4f}, Max: {dist.max():.4f}, Mean: {dist.mean():.4f}")
# 数据增强函数（保持原有逻辑）
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, test=False, device=None):
        super(NPY_datasets, self).__init__()
        self.train = train
        self.test = test

        # 加载图像和掩码路径
        data_type = 'train' if train else 'val'
        if test:
            data_type = 'test'
        else:
            data_type = 'train' if train else 'val'

        image_dir = os.path.join(path_Data, f'{data_type}/images')
        mask_dir = os.path.join(path_Data, f'{data_type}/masks')

        if not os.path.exists(image_dir):
            raise ValueError(f"图像目录不存在: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"掩码目录不存在: {mask_dir}")


        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        dist_dir = os.path.join(path_Data, f'{data_type}/distance_maps')  # 距离图目录
        if os.path.exists(dist_dir):
            self.dist_paths = sorted([os.path.join(dist_dir, f) for f in os.listdir(dist_dir)])
        else:
            print(f"警告：距离图目录不存在: {dist_dir}，将不使用距离图")
            self.dist_paths = []

        # 验证文件匹配
        self._validate_file_matching()


        # 统一输出尺寸 (256x256)
        self.output_size = (256, 256)

        # 验证集不需要数据增强，但需要固定尺寸
        self.val_transform = transforms.Compose([
            transforms.Resize(self.output_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_val_transform = transforms.Compose([
            transforms.Resize(self.output_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        # 训练集的随机增强类
        if self.train and not self.test:

            self.random_generator = RandomGenerator(self.output_size)

    def __getitem__(self, idx):
        img_path, msk_path = self.image_paths[idx], self.mask_paths[idx]
        dist_path = self.dist_paths[idx] if self.dist_paths else None
        # 获取文件名（带扩展名）
        name = os.path.basename(img_path)

        # 加载图像和掩码
        img = Image.open(img_path).convert('RGB')  # 确保 RGB 格式
        msk = Image.open(msk_path).convert('L')    # 确保灰度格式

        # 训练集：应用随机增强和尺寸调整
        if self.train and not self.test:

            sample = {'image': np.array(img), 'label': np.array(msk)}
            sample = self.random_generator(sample)
            img_tensor, msk_tensor = sample['image'], sample['label']
            # 加载并处理距离图
            if dist_path:
                dist_map = np.load(dist_path)
                # print(f"原始距离图范围: min {dist_map.min()}, max {dist_map.max()}")
                dist_map = Image.fromarray(dist_map).resize(self.output_size, Image.BILINEAR)
                dist_map = np.array(dist_map)
                # print(f"调整尺寸后距离图范围: min {dist_map.min()}, max {dist_map.max()}")
                if np.all(dist_map == 0):
                    print("警告：距离图全为零！")
                dist_tensor = torch.from_numpy(dist_map).float()
            else:
                dist_tensor = torch.zeros_like(msk_tensor).float()

        # 验证集：仅应用尺寸调整
        else:
            # 调整图像和掩码尺寸
            img = self.val_transform(img)
            msk = self.mask_val_transform(msk).squeeze(0).long()  # 确保掩码为 Long 类型
            img_tensor = img
            msk_tensor = msk
            # 验证集距离图
            if dist_path:
                dist_map = np.load(dist_path)
                dist_map = Image.fromarray(dist_map).resize(
                    self.output_size, Image.BILINEAR)
                dist_map = np.array(dist_map)
                dist_tensor = torch.from_numpy(dist_map).float()
            else:
                dist_tensor = torch.zeros_like(msk_tensor).float()
        # 数据合法性检查
        if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
            print(f"Invalid image at {img_path}")
        if torch.isnan(msk_tensor).any() or torch.isinf(msk_tensor).any():
            print(f"Invalid mask at {msk_path}")
        if torch.isnan(dist_tensor).any() or torch.isinf(dist_tensor).any():
            print(f"Invalid distance map at {dist_path if dist_path else 'generated'}")
        # 调试可视化：检查数据增强是否同步
        if hasattr(self, 'debug') and self.debug:
            debug_visualize(img_tensor, msk_tensor, dist_tensor, title="训练样本调试")
        return img_tensor, msk_tensor, dist_tensor
        # 返回 img, mask, distance_map, name
        # return img_tensor, msk_tensor, dist_tensor, name

    def _validate_file_matching(self):
        """适配带_segmentation后缀的文件名验证"""
        # 获取基础名（移除_segmentation后缀）
        image_bases = [os.path.splitext(os.path.basename(f))[0] for f in self.image_paths]
        mask_bases = [os.path.splitext(os.path.basename(f))[0].replace("_segmentation", "") for f in self.mask_paths]
        if self.dist_paths:
            dist_bases = [os.path.splitext(os.path.basename(f))[0].replace("_segmentation", "") for f in
                          self.dist_paths]
        else:
            dist_bases = []

            # 验证数量
        if self.dist_paths:
            assert len(image_bases) == len(mask_bases) == len(dist_bases), "文件数量不匹配"
        else:
            assert len(image_bases) == len(mask_bases), "图像和掩码文件数量不匹配"

            # 验证对应关系
        if self.dist_paths:
            for img, msk, dist in zip(image_bases, mask_bases, dist_bases):
                assert img == msk == dist, f"文件名不匹配: 图像({img}) vs 掩码({msk}) vs 距离图({dist})"
        else:
            for img, msk in zip(image_bases, mask_bases):
                assert img == msk, f"文件名不匹配: 图像({img}) vs 掩码({msk})"


    def __len__(self):
        return len(self.image_paths)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 随机旋转/翻转
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 转换为 PIL 图像以便调整尺寸
        image_pil = Image.fromarray(image)
        label_pil = Image.fromarray(label)

        # 调整尺寸（图像用双线性，掩码用最近邻）
        image_pil = image_pil.resize(self.output_size, Image.BILINEAR)
        label_pil = label_pil.resize(self.output_size, Image.NEAREST)

        # 转换回 NumPy 并归一化
        image = np.array(image_pil).astype(np.float32) / 255.0  # 归一化到 [0,1]
        label = np.array(label_pil).astype(np.float32)
        label = (label > 0.5).astype(np.float32)  # 二值化

        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label).unsqueeze(0).long()  # 添加通道维度并转为 Long

        return {'image': image, 'label': label}

# 测试代码
if __name__ == "__main__":
    path_Data = '../data/isic2017'  # 替换为实际路径
    config = None


    # 1. 实例化数据集
    try:
        train_dataset = NPY_datasets(path_Data, config, train=True)
        val_dataset = NPY_datasets(path_Data, config, train=False)
        print("数据集实例化成功！")
    except Exception as e:
        print(f"数据集实例化失败: {str(e)}")
        raise
    # train_dataset.debug = True
    # 2. 基础属性测试
    print(f"\n训练集数量: {len(train_dataset)} | 验证集数量: {len(val_dataset)}")
    print(f"示例图像路径: {train_dataset.image_paths[0]}")
    print(f"示例掩码路径: {train_dataset.mask_paths[0]}")
    if train_dataset.dist_paths:
        print(f"示例距离图路径: {train_dataset.dist_paths[0]}")

    # 3. 单样本测试
    sample_img, sample_msk, sample_dist = train_dataset[0]
    print("\n单样本测试:")
    print(
        f"图像 tensor shape: {sample_img.shape} | dtype: {sample_img.dtype} | range: [{sample_img.min()}, {sample_img.max()}]")
    print(
        f"掩码 tensor shape: {sample_msk.shape} | dtype: {sample_msk.dtype} | unique values: {torch.unique(sample_msk)}")
    print(
        f"距离图 tensor shape: {sample_dist.shape} | dtype: {sample_dist.dtype} | range: [{sample_dist.min()}, {sample_dist.max()}]")

    # 可视化单样本检查
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(sample_img.permute(1, 2, 0))  # CHW -> HWC
    plt.title("Image")
    plt.subplot(132)
    plt.imshow(sample_msk.squeeze(0), cmap='gray')
    plt.title("Mask")
    plt.subplot(133)
    plt.imshow(sample_dist, cmap='jet')
    plt.title("Distance Map")
    plt.tight_layout()
    plt.show()

    # 4. DataLoader 测试
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    print("\nBatch 测试:")
    for i, (img, msk, dist) in enumerate(train_loader):
        print(f"Train batch {i}:")
        print(f"  Images: {img.shape} | {img.dtype} | range: [{img.min()}, {img.max()}]")
        print(f"  Masks: {msk.shape} | {msk.dtype} | unique: {torch.unique(msk)}")
        print(f"  Distance Maps: {dist.shape} | {dist.dtype} | range: [{dist.min()}, {dist.max()}]")

        # 检查数据增强效果（仅对第一个 batch 可视化）
        if i == 0:
            fig, axes = plt.subplots(4, 3, figsize=(12, 16))
            for j in range(4):
                axes[j, 0].imshow(img[j].permute(1, 2, 0))
                axes[j, 0].set_title("Image")
                axes[j, 0].axis("off")
                axes[j, 1].imshow(msk[j].squeeze(0), cmap='gray')
                axes[j, 1].set_title("Mask")
                axes[j, 1].axis("off")
                axes[j, 2].imshow(dist[j], cmap='jet')
                axes[j, 2].set_title("Distance Map")
                axes[j, 2].axis("off")
            plt.tight_layout()
            plt.show()
        break  # 只测试第一个 batch

    # 验证集 DataLoader 测试
    for img, msk, dist in val_loader:
        print("\nVal batch:")
        print(f"  Images: {img.shape} | {img.dtype}")
        print(f"  Masks: {msk.shape} | {msk.dtype}")
        print(f"  Distance Maps: {dist.shape} | {dist.dtype}")
        break