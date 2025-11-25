import os
import torch
import numpy as np
import torch.utils.data as data
import cv2
import glob

def populate_train_list(images_path):
    input_list = glob.glob(os.path.abspath(images_path + '/Input/*'))
    ref_list = glob.glob(os.path.abspath(images_path + '/GT/*'))

    # ref_list = map(
    #     lambda x: x.replace('Input', 'GT'),input_list)
    
    return list(input_list), list(ref_list)

class lowlight_loader(data.Dataset):

    def __init__(self, images_path, height, width):
        
        self.input_list, self.ref_list = populate_train_list(images_path)
        self.input_list.sort()
        self.ref_list.sort()

        self.resize = True
        self.height = height
        self.width = width

        print("Total training examples:", len(self.input_list))


    def __getitem__(self, index):
        
        data_input = cv2.imread(self.input_list[index])
        data_ref = cv2.imread(self.ref_list[index])
        # data_input = cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)
        # data_ref = cv2.imread(self.ref_list[index], cv2.IMREAD_UNCHANGED)
        # cv2.imwrite('snapshots/result/_ori{}.jpg'.format(index),data_input)

        if data_input.shape[0] >= data_input.shape[1]:
            data_input = cv2.transpose(data_input)

        if data_ref.shape[0] >= data_ref.shape[1]:
            data_ref = cv2.transpose(data_ref)

        if self.resize:
            data_ref = cv2.resize(data_ref, (self.height, self.width))    # 512, 384 / 768, 576/ 1024, 768
            data_input = cv2.resize(data_input, (self.height, self.width))

        data_input_ori = data_input
        data_ref_ori = data_ref
        
        # cv2中的Lab范围都是0-255
        data_input = cv2.cvtColor(data_input, cv2.COLOR_BGR2RGB)
        data_ref = cv2.cvtColor(data_ref, cv2.COLOR_BGR2RGB)

        # data_input = (np.asarray(data_input[..., ::-1]) / 255.0)
        data_input = (np.asarray(data_input) / 255.0)
        data_input = torch.from_numpy(data_input).float()  # float32
        data_input = data_input.permute(2, 0, 1)
        # data_ref = (np.asarray(data_ref[..., ::-1]) / 255.0)
        data_ref = (np.asarray(data_ref) / 255.0)
        data_ref = torch.from_numpy(data_ref).float()
        data_ref = data_ref.permute(2, 0, 1)

        # test = data_ref.permute(1,2,0)
        # test = (test.numpy() * 255).astype(np.uint8)
        # test = cv2.cvtColor(test, cv2.COLOR_LAB2BGR)
        # cv2.imwrite('snapshots/result/_ori_{}.jpg'.format(index),test)

        return data_input, data_ref, data_input_ori, data_ref_ori, self.input_list[index], self.ref_list[index]
    
    def __len__(self):
        return len(self.input_list)