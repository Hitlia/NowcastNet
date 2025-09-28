import numpy as np
import os
import shutil
from torch.utils.data import Dataset,DataLoader
import torch
# import cv2
from scipy.signal import medfilt2d

class SinaDataset(Dataset):
    def __init__(self, input_param,split='all',time_window=1):
        self.data_path = input_param.data_path
        self.image_height = input_param.img_height
        self.image_width = input_param.img_width
        self.total_length = input_param.total_length
        self.input_data_type = input_param.input_data_type
        start_dict = {}
        start_npy = []
        kdp_npy = []
        zdr_npy = []
        kdp_path = self.data_path + '/KDP'
        zdr_path = self.data_path + '/ZDR'
        data_path = self.data_path + '/dBZ'

        name_list = os.listdir(data_path)
        name_list = sorted(name_list, key=lambda text:int(text[-3:]))
        len_name_list = len(name_list)
        split1 = int(len_name_list*7/10)
        split2 = int(len_name_list*9/10)
        if split == 'train':
            name_list = name_list[:split1]
        elif split == 'vali':
            name_list = name_list[split1:split2]
        elif split == 'test':
            name_list = name_list[split2:]
        elif split == 'mini':
            name_list = ['data_dir_027', 'data_dir_083', 'data_dir_022', 'data_dir_047', 'data_dir_028', 'data_dir_065', 'data_dir_098', 'data_dir_054']
       
        for data_dir in name_list:
            rain_path = data_path + str('/') + data_dir
            num_frame_rain = len(os.listdir(rain_path))
            kdp = kdp_path + str('/') + data_dir
            num_frame_kdp = len(os.listdir(kdp))
            zdr = zdr_path + str('/') + data_dir
            num_frame_zdr = len(os.listdir(zdr))
            num_frame = min(num_frame_rain,num_frame_kdp,num_frame_zdr)
            if num_frame < 20:
                continue
            start_npy.append(data_path + str('/') + data_dir+'/frame_'+str(0).zfill(3) + '.npy')
            kdp_npy.append(kdp_path + str('/') +data_dir+'/frame_'+str(0).zfill(3) + '.npy')
            zdr_npy.append(zdr_path + str('/') +data_dir+'/frame_'+str(0).zfill(3) + '.npy')
            for i in range(0,num_frame-20,time_window):
                start_npy.append(data_path + str('/') + data_dir+'/frame_'+str(i+time_window).zfill(3) + '.npy')
                kdp_npy.append(kdp_path + str('/') + data_dir+'/frame_'+str(i+time_window).zfill(3) + '.npy')
                zdr_npy.append(zdr_path + str('/') + data_dir+'/frame_'+str(i+time_window).zfill(3) + '.npy')
        start_dict['dBZ'] = start_npy
        start_dict['KDP'] = kdp_npy
        start_dict['ZDR'] = zdr_npy

        self.start_dict = start_dict
    
    def load(self,file_name,norm_param):
        data_frame = np.load(file_name)
        data_frame = medfilt2d(data_frame,kernel_size=3)
        norm_dict = {'dBZ': [0, 65],'ZDR': [-1, 5],'KDP': [-1, 6]}
        mmin, mmax = norm_dict[norm_param]
        data_frame = (data_frame-mmin) / (mmax - mmin)
        data_frame = data_frame.astype(self.input_data_type)
        return data_frame

    def get_dataset_data(self,index,data_name):
        self.start_npy = self.start_dict[data_name]
        initial_filename = self.start_npy[index]
        directory, filename = initial_filename.rsplit('/', maxsplit=1)
        prefix, extension = filename.rsplit('.', maxsplit=1)
        initial_frame_number = int(prefix[-3:])
        x_file_name = [initial_filename]
        y_file_name = []
        # 生成后续文件名并打印
        for i in range(1, 10):  # 生成帧号从1到20的文件名
            new_frame_number = initial_frame_number + i
            new_filename = f'{directory}/frame_{str(new_frame_number).zfill(3)}.{extension}'
            x_file_name.append(new_filename)
        for i in range(10,20):
            new_frame_number = initial_frame_number + i
            new_filename = f'{directory}/frame_{str(new_frame_number).zfill(3)}.{extension}'
            y_file_name.append(new_filename)
        
        array_shape = (self.total_length,self.image_width,self.image_height, 1)
        x_array = np.zeros(array_shape)
        y_array = np.zeros(array_shape)
        for i in range(self.total_length):
            x_array[i,:,:,0] = self.load(x_file_name[i],data_name)
            y_array[i,:,:,0] = self.load(y_file_name[i],data_name)
        return x_array,y_array
    
    def __getitem__(self,index):
        """读取npy中存储的图像"""
        dbz_x,dbz_y = self.get_dataset_data(index,'dBZ')
        kdp_x,kdp_y = self.get_dataset_data(index,'KDP')
        zdr_x,zdr_y = self.get_dataset_data(index,'ZDR')
        x = np.concatenate((dbz_x,kdp_x,zdr_x),axis=-1)
        y = np.concatenate((dbz_y,kdp_y,zdr_y),axis=-1)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x,y 

    def __len__(self):
        return len(self.start_dict['dBZ'])

def make_dataloaders(configs,splits=['train','vali','test']):
    dataloaders = []
    if 'train' in splits:
        train_dataset = SinaDataset(configs,'train')
        train_loader = DataLoader(train_dataset,batch_size=configs.batch_size,
                                shuffle=False,num_workers=configs.cpu_worker,drop_last=True)
        dataloaders.append(train_loader)
    if 'vali' in splits:
        vali_dataset = SinaDataset(configs,'vali')
        vali_loader = DataLoader(vali_dataset,batch_size=configs.batch_size,
                                shuffle=False,num_workers=configs.cpu_worker,drop_last=True)
        dataloaders.append(vali_loader)
    if 'test' in splits:
        test_dataset = SinaDataset(configs,'test')
        test_loader = DataLoader(test_dataset,batch_size=configs.batch_size,
                                shuffle=False,num_workers=configs.cpu_worker,drop_last=True)
        dataloaders.append(test_loader)
    return dataloaders

def make_dataloader(configs):
    sina_dataset = SinaDataset(configs)
    data_loader = DataLoader(sina_dataset,batch_size=configs.batch_size,
                            shuffle=False,num_workers=configs.cpu_worker,drop_last=True)
    return data_loader
        