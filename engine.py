import numpy as np
from sympy.printing.tests.test_codeprinter import test_print_Dummy
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from utils import save_imgs

import matplotlib.pyplot as plt
import numpy as np
import os
import random
def visualize_batch(images, preds, gts, distance_maps, save_dir=None, num_images=3, threshold=0.5):
    def to_numpy_and_squeeze(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        return np.squeeze(tensor)

    images = images.detach().cpu().numpy()  # [B, C, H, W]
    if preds is not None:
        preds = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    gts = gts.detach().cpu().numpy()
    distance_maps = distance_maps.detach().cpu().numpy()

    indices = random.sample(range(images.shape[0]), min(num_images, images.shape[0]))

    for idx in indices:
        img = images[idx].transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
        pred = to_numpy_and_squeeze(preds[idx])
        gt = to_numpy_and_squeeze(gts[idx])
        dist = to_numpy_and_squeeze(distance_maps[idx])

        pred_bin = (pred >= threshold).astype(np.uint8)
        gt_bin = (gt >= 0.5).astype(np.uint8)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(gt_bin, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(pred_bin, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(dist, cmap='jet')
        plt.title('Distance Map')
        plt.axis('off')

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'vis_{idx}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()

        images, targets, distance_maps = data
        images, targets, distance_maps = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float(), distance_maps.cuda(non_blocking=True).float()


        out = model(images)


        loss = criterion(out, targets, distance_maps)


        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    total_epochs = config.epochs
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        # for data in tqdm(test_loader):
        for step, data in enumerate(tqdm(test_loader)):
            img, msk, distance_maps = data
            img, msk, distance_maps = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), distance_maps.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk, distance_maps)
            if torch.isnan(loss):
                print("⚠️ Warning: Loss is NaN during validation!")
                print("Logits min:", out.min().item(), "max:", out.max().item())  # 检查输出范围
                print("Weights min:", distance_maps.min().item(), "max:", distance_maps.max().item())  # 检查权重范围
                print("Targets unique values:", torch.unique(msk))  # 检查标签值是否正常
                exit()  # 直接终止程序，避免继续运行

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)


    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)
        print(f'y_pre:{y_pre}, y_true:{y_true}')
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
        print(f'TN:{TN}, FP:{FP}, FN:{FN}, TP:{TP}')
        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)

def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None,
                   return_outputs=False):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):

            img, msk, distance_maps = data
            img, msk, distance_maps = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), distance_maps.cuda(non_blocking=True).float()


            out = model(img)

            loss = criterion(out, msk, distance_maps)


            loss_list.append(loss.item())
            msk_tensor = msk.detach()
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out_tensor = out.detach()
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if return_outputs and i < 600:  # 最多保存10张，防止占用内存
                results.append((
                    img.cpu()[0],  # [C, H, W], still Tensor
                    msk_tensor.cpu()[0].squeeze(0),              # 改为 [H, W]，避免变成 [256]
                    (out_tensor > 0.5).float().cpu()[0][0]  # HW
                ))
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy},specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    if return_outputs:
        return results
    else:
        return np.mean(loss_list)
