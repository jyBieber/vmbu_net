import torch
from torch.utils.data import DataLoader
from models.vmunet.vmbunet import VMUNet
from datasets.dataset import NPY_datasets
from engine import test_one_epoch
from utils import get_logger
from configs.config_setting import setting_config
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pandas as pd


def normalize_img(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.squeeze(x)

    if x.ndim == 1:
        raise ValueError(f"normalize_img 失败：输入维度为1维，shape={x.shape}")

    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)  # HWC
    elif x.ndim == 3 and x.shape[0] == 3:  # CHW -> HWC
        x = np.transpose(x, (1, 2, 0))

    if x.max() <= 1.0:
        x = (x * 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8)

    return x


def dice_score(pred, mask, threshold=0.5):
    """计算二值Dice得分"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    pred = np.squeeze(pred)
    mask = np.squeeze(mask)
    pred_bin = (pred >= threshold).astype(np.uint8)
    mask_bin = (mask >= 0.5).astype(np.uint8)

    intersection = np.sum(pred_bin * mask_bin)
    sum_ = np.sum(pred_bin) + np.sum(mask_bin)
    if sum_ == 0:
        return 1.0
    dice = 2 * intersection / sum_
    return dice


def iou_score(pred, mask, threshold=0.5):
    """计算二值mIoU"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    pred = np.squeeze(pred)
    mask = np.squeeze(mask)
    pred_bin = (pred >= threshold).astype(np.uint8)
    mask_bin = (mask >= 0.5).astype(np.uint8)

    intersection = np.sum(pred_bin * mask_bin)
    union = np.sum(pred_bin) + np.sum(mask_bin) - intersection
    if union == 0:
        return 1.0
    return intersection / union


def main():
    config = setting_config
    config.save_interval = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------------
    # 数据集加载
    # -----------------------------
    test_dataset = NPY_datasets(config.data_path, config, train=False, test=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -----------------------------
    # 模型加载
    # -----------------------------
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()
    model = model.to(device)

    # checkpoint_path = './results/vmunet_isic17_Sunday_02_November_2025_14h_45m_40s_BSF_bca_cbam_dist/checkpoints/best-epoch54-loss0.1820.pth'
    checkpoint_path = './results/vmunet_isic18_Tuesday_18_November_2025_09h_51m_03s/checkpoints/best-epoch181-loss0.2506.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # config.work_dir = './results/vmunet_isic17_Sunday_02_November_2025_14h_45m_40s_BSF_bca_cbam_dist/'
    config.work_dir = './results/vmunet_isic18_Tuesday_18_November_2025_09h_51m_03s/'

    filtered_state_dict = {
        k: v for k, v in checkpoint.items()
        if not (k.endswith('total_ops') or k.endswith('total_params') or 'total_ops' in k or 'total_params' in k)
    }
    model.load_state_dict(filtered_state_dict, strict=False)

    logger = get_logger('test_logger', 'logs')
    criterion = config.criterion

    # -----------------------------
    # 获取原始文件名
    # -----------------------------
    if hasattr(test_dataset, 'image_paths'):
        test_filenames = [os.path.splitext(os.path.basename(p))[0] for p in test_dataset.image_paths]
        print(f"从数据集 image_paths 获取了 {len(test_filenames)} 个文件名")
    else:
        print("警告：无法获取原始文件名，将使用索引作为文件名")
        test_filenames = [f"image_{i:04d}" for i in range(len(test_dataset))]

    # -----------------------------
    # 使用 test_one_epoch 函数并获取输出
    # -----------------------------
    print("🚀 开始推理...")

    # 调用 test_one_epoch 并获取返回的结果
    results = test_one_epoch(test_loader, model, criterion, logger, config,
                             test_data_name=None, return_outputs=True)

    # 检查返回结果的类型
    if isinstance(results, list):
        # 如果返回的是列表，说明有图像数据
        images, masks, preds = zip(*results)
        filenames = [test_filenames[i] for i in range(len(images))]
    else:
        # 如果返回的是单个值（如损失），则手动进行推理
        print("⚠️ test_one_epoch 返回了单个值，将手动进行推理...")
        model.eval()

        images_list = []
        masks_list = []
        preds_list = []

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader, desc="Testing")):
                # 解包三个值
                img, msk, distance_maps = data
                img = img.to(device)
                msk = msk.to(device)
                distance_maps = distance_maps.to(device)

                # 前向传播
                outputs = model(img)

                # 如果输出是元组，取第一个
                if type(outputs) is tuple:
                    outputs = outputs[0]

                # 保存结果
                images_list.append(img.cpu()[0])  # [C, H, W]
                masks_list.append(msk.cpu()[0].squeeze(0))  # [H, W]
                preds_list.append((outputs > 0.5).float().cpu()[0][0])  # [H, W]

        images = images_list
        masks = masks_list
        preds = preds_list
        filenames = test_filenames[:len(images)]

    # -----------------------------
    # 计算每张图的指标
    # -----------------------------
    all_records = []
    print("📊 正在计算每张图的 Dice 和 mIoU ...")
    for name, pred, mask in tqdm(zip(filenames, preds, masks), total=len(filenames)):
        try:
            dice_val = dice_score(pred, mask)
            miou_val = iou_score(pred, mask)
            all_records.append({
                "filename": name,
                "Dice": dice_val,
                "mIoU": miou_val
            })
        except Exception as e:
            print(f"[跳过] {name} 计算失败: {e}")

    # -----------------------------
    # 保存到 CSV 文件
    # -----------------------------
    csv_path = os.path.join(config.work_dir, "fanhua_val_metrics.csv")
    pd.DataFrame(all_records).to_csv(csv_path, index=False)
    print(f"✅ 所有结果已保存到 {csv_path}")

    # -----------------------------
    # 保存可视化结果
    # -----------------------------
    save_dir = os.path.join(config.work_dir, 'fanhua_val_results')
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 保存预测图像到 {save_dir} ...")

    for name, img, mask, pred in tqdm(zip(filenames, images, masks, preds), total=len(filenames)):
        img = normalize_img(img)
        mask = normalize_img(mask)
        pred = normalize_img(pred)
        concat = np.concatenate([img, mask, pred], axis=1)
        Image.fromarray(concat).save(os.path.join(save_dir, f"{name}.png"))

    print("🎯 测试完成！")


if __name__ == '__main__':
    main()