import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
random.seed(777)
# gpus = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import torch
import cv2
import time
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
# import model
from model import enc_net
from torchvision import transforms, utils
import dataloader
import util

# loss function
from torchvision.models import vgg16

# Salience
from models import U2NETP # small version u2net 4.7 MB
from fastsam import FastSAM, FastSAMPrompt
# torch.cuda.set_device(1)

def train(config):

    # Salience init
    model_name='u2netp'#u2netp
    print("...load U2NEP---4.7 MB")
    net = U2NETP(3,1)
    model_dir = os.path.join(os.getcwd(), './checkpoints/', model_name + '.pth')
    sal_transform = transforms.Compose([
                transforms.Resize((320,320), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        # net = nn.DataParallel(net)
        net.cuda()
        # print("Gpu")
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        # print("Cpu")
    net.eval()
    # load the sam model
    seg_model_path = './checkpoints/FastSAM-x.pt'
    model_sam = FastSAM(seg_model_path)
    DEVICE = torch.device("cuda")

    # Load model
    lab_net = enc_net().cuda()
    # lab_net = nn.DataParallel(lab_net)
    if config.load_pretrain == True:
        lab_net.load_state_dict(torch.load(config.pretrain_dir))
    
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    ## DataLoaders
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path, config.image_height, config.image_width)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                        shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    
    # set the val loader: batchsize->1;heught->768;width->576
    # val_dataset = dataloader.lowlight_loader(config.val_images_path, config.image_height, config.image_width)
    val_dataset = dataloader.lowlight_loader(config.val_images_path, config.val_height, config.val_width)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size,
                        shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=True)
	
	# loss
    # L_L1 = nn.L1Loss()
    L_L1 = nn.SmoothL1Loss()
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = util.LossNetwork(vgg_model)
    loss_network.eval()
    # L_spa = util.L_spa()
    # L_TV = util.L_TV()
    # L_char = util.CharbonnierLoss()
    L_edge = util.EdgeLoss()

    ## Optimizer
    optimizer = torch.optim.Adam(lab_net.parameters(), 
                    lr=config.lr, weight_decay=config.weight_decay)
    
    if len(device_ids)>1:
        lab_net = nn.DataParallel(lab_net, device_ids = device_ids)

    ## Tensorboard
    writer = SummaryWriter('./logs_c')

    # Start training!
    print('==> Training start: ')
    best_psnr = 0
    best_ssim = 0
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    for epoch in range(config.num_epochs):

        lab_net.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        img_idx = 0
        # for iteration, img_lowlight in enumerate(tqdm(train_loader)):
        for iteration, img_lowlight in enumerate(train_loader):

            img_input, img_ref, img_input_ori, img_ref_ori, input_path, ref_path = img_lowlight
            img_input_ori = img_input_ori.detach().numpy()
            img_ref_ori = img_ref_ori.detach().numpy()
            img_seg = []
            for i in range(img_input_ori.shape[0]):
                img_seg.append(img_input_ori[i])
            # test val
            # break
            # img_seg.append(img_input_ori[i] for i in range(img_input_ori.shape[0]))
            # img_seg.append(img_ref_ori[i] for i in range(img_ref_ori.shape[0]))
            # print(input_path)
            masks_sam = model_sam(img_seg, device=DEVICE, 
                                  retina_masks=True, imgsz=1024,conf=0.4,iou=0.9,)
            img_input = img_input.cuda()
            img_ref = img_ref.cuda()

            # Salience
            img_sal = sal_transform(img_input)
            d,_,_,_,_,_,_ = net(img_sal)
            d = F.interpolate(util.normPRED(d[:,0,:,:]).unsqueeze(1), size=(config.image_width, config.image_height), mode='bilinear', align_corners=True)
            # produce hot map
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
            # d_ = ax1.imshow(d[0,0,:,:].cpu().detach().numpy(),cmap='hot', interpolation='nearest')
            # pre_ = ax2.imshow(pre[0,0,:,:].cpu().detach().numpy(),cmap='hot', interpolation='nearest')
            # # fig.colorbar(d_)
            # # fig.colorbar(pre_)
            # fig.savefig('./snapshots/heatmap.png')
            masks = util.get_mask(masks_sam, d)
            masks_sal = masks[0].unsqueeze(0)
            for id in range(len(masks)):

                # branch for global
                # if masks[id].sum()/(masks[id].shape[1]*masks[id].shape[2]) < 0.050:
                #      masks[id].fill_(False)
                    #  print(input_path[id])
                if id > 0:
                        masks_sal = torch.cat((masks_sal, masks[id].unsqueeze(0)), dim=0)
            # print(masks_sal.shape)
            # masks_sal = torch.cat((masks[0].unsqueeze(0),masks[1].unsqueeze(0)), dim=0)
            # 检验提取效果
            # sal_demo_fore = img_ref[0] * masks_sal[0]
            # sal_demo_back = img_ref[0] * ~masks_sal[0]
            # sal_demo_fore = transforms.functional.to_pil_image(sal_demo_fore)
            # sal_demo_back = transforms.functional.to_pil_image(sal_demo_back)
            # sal_demo_fore.save('./snapshots/sam_fore.png')
            # sal_demo_back.save('./snapshots/sam_back.png')
            # u2_seg = d > 0.008
            # sal_demo_u2 = img_ref[0] * u2_seg[0]
            # sal_demo_u2_save = transforms.functional.to_pil_image(sal_demo_u2)
            # sal_demo_u2_save.save('./snapshots/u2_fore.png')

            # model add img_input
            enhance_sal, enhance_bg, enhance_mid, enhance_img = lab_net(masks_sal, img_input)
            # enhanced = enhance_sal + enhance_bg * ~mask
            # enhance_img = enhance_sal + enhance_bk
            # enhance_bk_sal = enhance_bk * mask

            # loss_sal = ((loss_network(enhance_sal, img_ref*pre) / 5 + L_L1(enhance_sal, img_ref*pre)))*1.5
            # loss_bg = loss_network(enhance_bg*(1-pre), img_ref*(1-pre)) / 5 + L_L1(enhance_bg*(1-pre), img_ref*(1-pre))
            loss_mid = loss_network(enhance_mid, img_ref) / 5 + L_L1(enhance_mid, img_ref) * 3
            loss_edge = torch.sum(torch.stack([L_edge(enhance_img[j],img_ref) for j in range(len(enhance_img))]))
            # loss_ful = loss_network(enhance_img[0], img_ref) / 5 + L_L1(enhance_img[0], img_ref)*3
            loss_ful = L_L1(enhance_img[0], img_ref)
            # loss = loss_mid + loss_edge/(loss_edge/loss_mid).detach() + loss_ful/(loss_ful/loss_mid).detach()
            # loss = loss_edge/loss_edge.detach() + loss_ful/loss_ful.detach()
            # loss = loss_mid/loss_mid.detach() + loss_edge/loss_edge.detach() + loss_ful/loss_ful.detach()
            loss = loss_mid*loss_mid.detach() + loss_edge*loss_edge.detach() + loss_ful*loss_ful.detach()
            writer.add_scalar('loss_mid', loss_mid.item(), iteration)
            writer.add_scalar('loss_edge', loss_edge.item(), iteration)
            writer.add_scalar('loss_ful', loss_ful.item(), iteration)
            writer.add_scalar('loss', loss.item(), iteration)
            # loss = loss_mid^2 + loss_edge^2 + loss_ful^2
            # loss = loss_sal_l1 + loss_bg_l1 + loss_char + loss_edge

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lab_net.parameters(),config.grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item()

            if iteration % 90 == 0 and epoch % 4 == 0:
                    # print("iteration: %d, loss: %f, loss_ful: %f, loss_edge: %f" % (iteration, loss.item(), loss_ful.item(), loss_edge.item()))
                    print("iteration: %d, loss: %f, loss_ful: %f, loss_mid: %f, loss_edge: %f" % (iteration, loss.item(), loss_ful.item(), loss_mid.item(), loss_edge.item()))
                    for i in range(img_input.shape[0]):
                        result_compare = torch.cat([img_input[i], enhance_img[0][i], img_ref[i]], dim=2)
                        utils.save_image(result_compare, '{}single_GPU/{}_{}_{}_compare.png'.format(
                            config.snapshots_folder, epoch, iteration, input_path[i].split('/')[-1].split('.')[0]))
                        # utils.save_image(img_input[i], '{}result_k10_2/{}_{}_{}_ori.png'.format(
                        #     config.snapshots_folder, epoch, iteration, img_idx))
                        utils.save_image(enhance_img[0][i], '{}single_GPU/{}_{}_{}_enh.png'.format(
                            config.snapshots_folder, epoch, iteration, input_path[i].split('/')[-1].split('.')[0]))
                        # utils.save_image(img_ref[i], '{}result_k10_2/{}_{}_{}_gt.png'.format(
                        #     config.snapshots_folder, epoch, iteration, img_idx))
                        # utils.save_image(enhance_bg[i], '{}result_k10_2/{}_{}_{}_bk.png'.format(
                        #     config.snapshots_folder, epoch, iteration, img_idx))
                        # utils.save_image(enhance_sal[i], '{}result_k10_2/{}_{}_{}_sal.png'.format(
                        #     config.snapshots_folder, epoch, iteration, img_idx))
                        # utils.save_image(enhance_mid[i], '{}result_k10_2/{}_{}_{}_mid.png'.format(
                        #     config.snapshots_folder, epoch, iteration, img_idx))
                        img_idx += 1

        if epoch % config.val_epochs == 0:

            lab_net.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            for ii, data_val in enumerate(val_loader):
                img_input, img_ref, img_input_ori, img_ref_ori, input_path, _ = data_val
                img_input_ori = img_input_ori.detach().numpy()
                # img_ref_ori = img_ref_ori.detach().numpy()
                img_seg = []
                for i in range(img_input_ori.shape[0]):
                    img_seg.append(img_input_ori[i])
                # print(input_path)
                masks_sam = model_sam(img_seg, device=DEVICE, 
                                  retina_masks=True, imgsz=1024,conf=0.4,iou=0.9,)
                img_input = img_input.cuda()
                img_ref = img_ref.cuda()

                # Salience
                img_sal = sal_transform(img_input)
                d,_,_,_,_,_,_= net(img_sal)
                # pred_p = F.interpolate(util.normPRED(d[:,0,:,:]).unsqueeze(1), size=(384, 512), mode='bilinear', align_corners=True)
                # mask = F.interpolate(util.normPRED(d[:,0,:,:]).unsqueeze(1), size=(384, 512), mode='bilinear', align_corners=True) > 0.008
                
                # set the u2 seg result size
                # d = F.interpolate(util.normPRED(d[:,0,:,:]).unsqueeze(1), size=(config.image_width, config.image_height), mode='bilinear', align_corners=True)
                d = F.interpolate(util.normPRED(d[:,0,:,:]).unsqueeze(1), size=(config.val_width, config.val_height), mode='bilinear', align_corners=True)
                masks = util.get_mask(masks_sam, d)
                masks_sal = masks[0].unsqueeze(0)
                for id in range(len(masks)):

                    # branch for global
                    # if masks[id].sum()/(masks[id].shape[1]*masks[id].shape[2]) < 0.070:
                    #     masks[id].fill_(False)
                    #     print(input_path[id])
                    if id > 0:
                        masks_sal = torch.cat((masks_sal, masks[id].unsqueeze(0)), dim=0)
                # masks_sal = torch.cat((masks[0].unsqueeze(0),masks[1].unsqueeze(0)), dim=0)
                # mask = d > 0.1	# Original: 0.008
                # pre = torch.sigmoid(10*(d-0.5))
                # pre = torch.clamp(pre, min=1e-2, max=1.0)

                # 检验提取效果
                # sal_demo_fore = img_ref[0] * masks_sal[0]
                # sal_demo_back = img_ref[0] * ~masks_sal[0]
                # sal_demo_fore = transforms.functional.to_pil_image(sal_demo_fore)
                # sal_demo_back = transforms.functional.to_pil_image(sal_demo_back)
                # sal_demo_fore.save('./snapshots/sam_fore.png')
                # sal_demo_back.save('./snapshots/sam_back.png')
                # u2_seg = d > 0.008
                # sal_demo_u2 = img_ref[0] * u2_seg[0]
                # sal_demo_u2_save = transforms.functional.to_pil_image(sal_demo_u2)
                # sal_demo_u2_save.save('./snapshots/u2_fore.png')

                with torch.no_grad():
                    _, _, _, enhance_img = lab_net(masks_sal, img_input)
                    # enhance_img = enhance_sal + enhance_bk
                # for input, ref in zip(enhance_img, sal_ref):
                psnr_val_rgb.append(util.torchPSNR(enhance_img[0], img_ref))
                ssim_val_rgb.append(util.torchSSIM(enhance_img[0], img_ref))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

            # add to tensorboard
            writer.add_scalar('psnr_val', psnr_val_rgb, epoch)
            writer.add_scalar('ssim_val', ssim_val_rgb, epoch)
            # Save the best PSNR model of validation
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch_psnr = epoch
                torch.save(lab_net.state_dict(), config.checkpoints + "bestPSNR_Epoch" + str(best_epoch_psnr) + '.pth')
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
                epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

            # Save the best SSIM model of validation
            if ssim_val_rgb > best_ssim:
                best_ssim = ssim_val_rgb
                best_epoch_ssim = epoch
                torch.save(lab_net.state_dict(), config.checkpoints + "bestSSIM_Epoch" + str(best_epoch_ssim) + '.pth')
            print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
                epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))
            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\t".format(epoch, time.time() - epoch_start_time,
                                                                                        epoch_loss))
            print("------------------------------------------------------------------")
    writer.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="/home/x3022/ChenPeng/Dataset/tiny_sal/train")
    parser.add_argument('--val_images_path', type=str, default="/home/x3022/ChenPeng/Dataset/tiny_sal/val")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=320)
    parser.add_argument('--val_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=18)
    parser.add_argument('--display_iter', type=int, default=50)
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--image_height', type=int, default=400)	# 768/512/384/480/384/400
    parser.add_argument('--image_width', type=int, default=320)		# 512/384/256/320/288/320
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
    parser.add_argument('--checkpoints', type=str, default="checkpoints/weights_singleGPU/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "checkpoints/bestSSIM_Epoch152.pth")
    parser.add_argument('--val_height', type=int, default=400)	
    parser.add_argument('--val_width', type=int, default=320)	#768/576

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train(config)

