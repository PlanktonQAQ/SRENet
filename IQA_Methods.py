import util
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pyiqa
from PIL import Image
import numpy as np
import cv2
from natsort import natsorted
from glob import glob

## path of the result
# our:/home/x3022/ChenPeng/Code/New_Sal/snapshots/result/result_3_152
# MIRNet:/home/x3022/ChenPeng/Download/MIR/test_MIR
# HWMNet:/home/x3022/ChenPeng/Download/HWMNet/FiveK_test
# IAT:/home/x3022/ChenPeng/Download/IAT/Result_IAT_v1
# LLFormer:/home/x3022/ChenPeng/Download/LLFormer/tiny_sal_test
# PairLIE:/home/x3022/ChenPeng/Download/PairLIE/FiveK/I
# Retinexformer:/home/x3022/ChenPeng/Download/RetinexFormer_FiveK/FiveK
# SCI:/home/x3022/ChenPeng/Download/SCI/tiny_sal_test
# LLFlow_SKF:/home/x3022/ChenPeng/Download/LLFlow_SKF/000
# DCCNet:/home/x3022/ChenPeng/Code/DCC-Net-main/results/FiveK

# target_path = '/home/x3022/ChenPeng/Code/New_Sal/snapshots/result/result_3_152'
target_path = '/home/x3022/ChenPeng/Code/New_Sal/snapshots/result_F'
gt_path = '/home/x3022/ChenPeng/Dataset/tiny_sal/test/GT'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# target_list = os.listdir(target_path)
# gt_list = os.listdir(gt_path)

target_list = natsorted(glob(os.path.join(target_path, '*.jpg'))
                  + glob(os.path.join(target_path, '*.png')))
gt_list = natsorted(glob(os.path.join(gt_path, '*.jpg'))
                  + glob(os.path.join(gt_path, '*.png')))
if len(target_list) == 0:
    raise Exception(f"No files found at {target_path}")
if len(gt_list) == 0:
    raise Exception(f"No files found at {gt_path}")

## list all available metrics
print(pyiqa.list_models())

psnr_val_rgb = []
ssim_val_rgb = []
lpips_val_rgb = []
# no reference IQA
brisque_val_rgb = []
niqe_val_rgb = []
ilniqe_val_rgb = []
dbcnn_val_rgb = []
musiq_koniq_val_rgb = []
clipiqa2_val_rgb = []
# dists_val_rgb = []
# fsim_val_rgb = []
width = 576
height = 768

## use the pyiqa
## create metric with default setting
iqa_metric = pyiqa.create_metric('lpips', device=device)
brisque_nf_metric = pyiqa.create_metric('brisque', device=device)
niqe_nf_metric = pyiqa.create_metric('niqe', device=device)
ilniqe_nf_metric = pyiqa.create_metric('ilniqe', device=device)
dbcnn_nf_metric = pyiqa.create_metric('dbcnn', device=device)
musiq_koniq_nf_metric = pyiqa.create_metric('musiq-koniq', device=device)
clipiqa2_nf_metric = pyiqa.create_metric('clipiqa+', device=device)

for i in range(len(target_list)):
    img_target = Image.open(target_list[i]).convert('RGB')
    img_gt = Image.open(gt_list[i]).convert('RGB')
    if img_target.size[0] < img_target.size[1]:
        img_target = img_target.transpose(Image.ROTATE_90)
    if img_gt.size[0] < img_gt.size[1]:
        img_gt = img_gt.transpose(Image.ROTATE_90)
    # if img_target.size[0] != height or img_target.size[1] != width:
    #     img_target = img_target.resize((height, width), Image.BILINEAR)
        # img_target.save('./snapshots/target.png')
    # if img_gt.size[0] != height or img_gt.size[1] != width:
    #     img_gt = img_gt.resize((height, width), Image.BILINEAR)

    if img_gt.size[0] != img_target.size[0] or img_gt.size[1] != img_target.size[1]:
        img_gt = img_gt.resize((img_target.size[0], img_target.size[1]), Image.BILINEAR)
        # img_gt.save('./snapshots/gt.png')
    # img_target.save('/home/x3022/ChenPeng/Code/New_Sal/snapshots/test_IQA/{}_target.png'.format(i))
    # img_gt.save('/home/x3022/ChenPeng/Code/New_Sal/snapshots/test_IQA/{}_gt.png'.format(i))
    
    # transform PIL to numpy
    img_target = np.array(img_target)
    img_gt = np.array(img_gt)
    # transform numpy to tensor
    img_target = torch.from_numpy(img_target/255).float()
    img_gt = torch.from_numpy(img_gt/255).float()
    img_target = img_target.permute(2, 0, 1).unsqueeze(0)
    img_gt = img_gt.permute(2, 0, 1).unsqueeze(0)
    
    # caculate PSNR and SSIM
    psnr_val_rgb.append(util.torchPSNR(img_target, img_gt))
    ssim_val_rgb.append(util.torchSSIM(img_target, img_gt))
    print('Target_Path:{}, GT:{}'.format(target_list[i].split('/')[-1], gt_list[i].split('/')[-1]))

    # dists_fr_metric = pyiqa.create_metric('dists', device=device)
    # fsim_fr_metric = pyiqa.create_metric('fsim', device=device)
    ## check if lower better or higher better
    # print(iqa_metric.lower_better)
    
    ## example for iqa score inference
    ## Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
    score_fr = iqa_metric(img_target, img_gt)
    score_brisque_nr = brisque_nf_metric(img_target)
    score_niqe_nr = niqe_nf_metric(img_target)
    score_ilniqe_nr = ilniqe_nf_metric(img_target)
    score_dbcnn_nr = dbcnn_nf_metric(img_target)
    score_musiq_koniq_nr = musiq_koniq_nf_metric(img_target)
    score_clipiqa2_nr = clipiqa2_nf_metric(img_target)
    # score_dists_fr = dists_fr_metric(img_target, img_gt)
    # score_fsim_fr = fsim_fr_metric(img_target, img_gt)
    # score_nr = iqa_metric(img_tensor_x)
    lpips_val_rgb.append(score_fr)
    brisque_val_rgb.append(score_brisque_nr)
    niqe_val_rgb.append(score_niqe_nr)
    ilniqe_val_rgb.append(score_ilniqe_nr)
    dbcnn_val_rgb.append(score_dbcnn_nr)
    musiq_koniq_val_rgb.append(score_musiq_koniq_nr)
    clipiqa2_val_rgb.append(score_clipiqa2_nr)
    # dists_val_rgb.append(score_dists_fr)
    # fsim_val_rgb.append(score_fsim_fr)

    print('PSNR: {:.4f} dB, SSIM: {:.4f}, LPIPS: {}'.format(psnr_val_rgb[i], ssim_val_rgb[i], lpips_val_rgb[i]))

    ## img path as inputs.
    # score_fr = iqa_metric('./ResultsCalibra/dist_dir/I03.bmp', './ResultsCalibra/ref_dir/I03.bmp')


psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()
lpips_val_rgb = torch.stack(lpips_val_rgb).mean().item()

brisque_val_rgb = torch.stack(brisque_val_rgb).mean().item()
niqe_val_rgb = torch.stack(niqe_val_rgb).mean().item()
ilniqe_val_rgb = torch.stack(ilniqe_val_rgb).mean().item()
dbcnn_val_rgb = torch.stack(dbcnn_val_rgb).mean().item()
musiq_koniq_val_rgb = torch.stack(musiq_koniq_val_rgb).mean().item()
clipiqa2_val_rgb = torch.stack(clipiqa2_val_rgb).mean().item()
# dists_val_rgb = torch.stack(dists_val_rgb).mean().item()
# fsim_val_rgb = torch.stack(fsim_val_rgb).mean().item()
print('PSNR_Mean: {:.4f} dB, SSIM_Mean: {:.4f}, LPIPS_Mean: {}'.format(psnr_val_rgb, ssim_val_rgb, lpips_val_rgb))
# print('DISTS_Mean: {}, FSIM_Mean: {}'.format(dists_val_rgb, fsim_val_rgb))
print('BRISQUE_Mean: {}, NIQE_Mean: {}, ILNIQE_Mean: {}, DBCNN_Mean: {}, MUSIQ_KONIQ_Mean: {}, CLIPIQA2_Mean: {}'.format(brisque_val_rgb, niqe_val_rgb, ilniqe_val_rgb, dbcnn_val_rgb, musiq_koniq_val_rgb, clipiqa2_val_rgb))

# For FID metric, use directory or precomputed statistics as inputs
# refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
# fid_metric = pyiqa.create_metric('fid')
# score = fid_metric('./ResultsCalibra/dist_dir/', './ResultsCalibra/ref_dir')
# score = fid_metric('./ResultsCalibra/dist_dir/', dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")