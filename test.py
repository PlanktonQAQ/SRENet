import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from natsort import natsorted
from glob import glob
from PIL import Image
# import model
from model import enc_net
# from model3 import r_net
from torchvision import transforms, utils
import util

# Salience
from models import U2NETP # small version u2net 4.7 MB
from fastsam import FastSAM, FastSAMPrompt

# torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
# yolo file path:/media/x3022/42B0CAB7B0CAB0A9/PengChen/Datasets/Yolo_Enhancement/Input/
parser.add_argument('--filepath', type=str, default='/home/x3022/ChenPeng/Dataset/tiny_sal/test/Input/')
parser.add_argument('--result_dir', type=str, default='./snapshots/result_F/')
parser.add_argument('--pretrain_path', type=str, default='./checkpoints/bestSSIM.pth')
parser.add_argument('--image_height', type=int, default=768)    # 576/768
parser.add_argument('--image_width', type=int, default=576)     # 512/768
args = parser.parse_args()

inp_dir = args.filepath
out_dir = args.result_dir
os.makedirs(out_dir, exist_ok=True)

# Salience init
model_name='u2netp'
print("...load U2NEP---4.7 MB")
net = U2NETP(3,1)
model_dir = './checkpoints/u2netp.pth'
sal_transform = transforms.Compose([
            transforms.Resize((320,320), interpolation=2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
net.eval()


files = natsorted(glob(os.path.join(inp_dir, '*P1.5.JPG')) +
                  glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.JPEG')))
if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load model and weights
lab_net = enc_net().cuda()
device_ids = [i for i in range(torch.cuda.device_count())]

if len(device_ids)>1:
        lab_net = nn.DataParallel(lab_net, device_ids = device_ids)

lab_net.load_state_dict(torch.load(args.pretrain_path))
lab_net.eval()
 # load the sam model
seg_model_path = './checkpoints/FastSAM-x.pt'
model_sam = FastSAM(seg_model_path)
DEVICE = torch.device("cuda")
print('restoring images......')

mul = 8
index = 0
for file_ in files:
    
    # file_gt = file_.replace('Input', 'GT')
    # img_gt = Image.open(file_gt).convert('RGB')
    img = Image.open(file_).convert('RGB')
    # print(file_)

    # Normal
    if img.size[0] < img.size[1]:
        # img_gt = img_gt.transpose(Image.ROTATE_90)
        img = img.transpose(Image.ROTATE_90)
    # img_gt = img_gt.resize((args.image_height, args.image_width), Image.BILINEAR)
    img = img.resize((args.image_height, args.image_width), Image.BILINEAR)

    # for yolo detection
    height = img.size[1]
    width = img.size[0]
    # if img.size[0] > 2000 or img.size[1] > 2000:
    #     height = img.size[1] // 2
    #     width = img.size[0] // 2
    #     img = img.resize((width,height), Image.BILINEAR)
    # if img.size[0] > 1600 or img.size[1] > 1600:
    #     height = img.size[1] // 2
    #     width = img.size[0] // 2
    #     img = img.resize((width,height), Image.BILINEAR)
    # if img.size[0] % 8 != 0 or img.size[1] % 8 != 0:
    #     height = img.size[1] // mul * mul
    #     width = img.size[0] // mul * mul
    #     img = img.resize((width,height), Image.BILINEAR)
    # use sam to seg the image
    masks_sam = model_sam(img, device=DEVICE, 
                                  retina_masks=True, imgsz=1024,conf=0.4,iou=0.9,)
    # transform the image
    # img_gt = transforms.functional.to_tensor(img_gt).unsqueeze(0).cuda()
    input_ = transforms.functional.to_tensor(img).unsqueeze(0).cuda()

    # Salience
    img_sal = sal_transform(input_)
    d,_,_,_,_,_,_= net(img_sal)
    d = F.interpolate(util.normPRED(d[:,0,:,:]).unsqueeze(1), size=(height, width), mode='bilinear', align_corners=True)
    # d = F.interpolate(util.normPRED(d[:,0,:,:]).unsqueeze(1), size=(args.image_width, args.image_height), mode='bilinear', align_corners=True)
    masks = util.get_mask(masks_sam, d)
    # masks_sal = masks[0].unsqueeze(0)
    for id in range(len(masks)):
        if masks[id].sum()/(masks[id].shape[1]*masks[id].shape[2]) < 0.03:
            masks[id].fill_(False)
            print(file_)
    # masks = masks[0].unsqueeze(0)
    masks_u2 = d > 0.08
    masks_sal = masks[0].unsqueeze(0)

    # for id in range(len(masks)):

    ##     branch for global
    #     if masks[id].sum()/(masks[id].shape[1]*masks[id].shape[2]) < 0.050:
    #          masks[id].fill_(False)
    #          print(input_path[id])
    #     if id > 0:
    #             masks_sal = torch.cat((masks_sal, masks[id].unsqueeze(0)), dim=0)

    # Enhance
    with torch.no_grad():
        enhance_sal, enhance_bg, enhance_mid, enhance_img = lab_net(masks_sal, input_)
    f = os.path.splitext(os.path.split(file_)[-1])[0]
    input_sal = input_[0] * masks_sal[0]
    input_bg = input_[0] - input_sal
    enhance_sal = enhance_img[0][0] * masks_sal[0]
    enhance_bg = enhance_img[0][0] - enhance_sal

    input_sal_u2 = input_[0] * masks_u2[0]
    input_bg_u2 = input_[0] - input_sal_u2
    enhance_sal_u2 = enhance_img[0][0] * masks_u2[0]
    enhance_bg_u2 = enhance_img[0][0] - enhance_sal_u2
    # result_compare = torch.cat([input_, enhance_img[0], img_gt], dim=3)
    # utils.save_image(result_compare, (os.path.join(out_dir, f + '.png')))
    
    # separate of the image
    # utils.save_image(enhance_mid, (os.path.join(out_dir, f + '_mid.png')))
    utils.save_image(enhance_img[0], (os.path.join(out_dir, f + '.png')))
    # utils.save_image(input_sal, (os.path.join(out_dir, f + '_in_sal.png')))
    # utils.save_image(input_bg, (os.path.join(out_dir, f + '_in_bg.png')))
    # utils.save_image(enhance_sal, (os.path.join(out_dir, f + '_en_sal.png')))
    # utils.save_image(enhance_bg, (os.path.join(out_dir, f + '_en_bg.png')))
    # utils.save_image(input_sal_u2, (os.path.join(out_dir, f + '_in_sal_u2.png')))
    # utils.save_image(input_bg_u2, (os.path.join(out_dir, f + '_in_bg_u2.png')))
    # util.save_image(enhance_sal_u2, (os.path.join(out_dir, f + '_en_sal_u2.png')))
    # utils.save_image(enhance_bg_u2, (os.path.join(out_dir, f + '_en_bg_u2.png')))

    # utils.save_image(enhance_sal, (os.path.join(out_dir, f + '_sal.png')))
    # utils.save_image(enhance_bg, (os.path.join(out_dir, f + '_bg.png')))
    # utils.save_image(img_gt, (os.path.join(out_dir, f + '_gt.png')))
    index += 1
    print('%d/%d' % (index, len(files)))

    # FastSAM
    # prompt_process = FastSAMPrompt(file_, masks_sam, device=DEVICE)
    # ann = prompt_process.everything_prompt()
    # # sam_path = os.path.join('/home/x3022/ChenPeng/Code/New_Sal/snapshots/result/result_single_square_test_127_pipeline', f + '_sam.png')
    # prompt_process.plot(annotations=ann,output='./snapshots/result/result_single_square_final_218_pipeline/',)


print(f"Files saved at {out_dir}")
print('finish !')

