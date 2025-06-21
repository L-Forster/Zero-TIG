import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import re


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def initialize(self, args, task):
        pass

    def extract_number(self, filename):
        match = os.path.splitext(os.path.split(filename)[1])[0]
        return int(match) if match else 0  # Extract the numeric part and convert it to an integer

    def sort_files_by_name(self, img_list):
        # Sort the file
        sorted_files = sorted(img_list, key=self.extract_number)
        return sorted_files


class DefaultDataset(BaseDataset):
    def initialize(self, args, task):
        self.args = args
        self.low_img_dir = args.lowlight_images_path
        self.task = task
        self.train_low_data_names = []
        self.train_target_data_names = []
        assert os.path.exists(self.low_img_dir), "Input directory does not exist!"

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                if '.' == name[0]:
                    continue
                self.train_low_data_names.append(os.path.join(root, name))

        # self.train_low_data_names.sort()
        self.train_low_data_names = self.sort_files_by_name(self.train_low_data_names)
        # if 'test' == task:
        #     self.train_low_data_names = self.train_low_data_names[:10]

        self.count = len(self.train_low_data_names)
        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.last_data_name_path = self.train_low_data_names[0]

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        new_size = (1920, 1080)
        im = im.resize(new_size)
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):
        img_path = self.train_low_data_names[index]
        ll = self.load_images_transform(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        last_data_name_path = self.last_data_name_path
        self.last_data_name_path = img_path
        return ll, img_name, img_path, last_data_name_path
    
    def __len__(self):
        return self.count

    def name(self):
        return 'DefaultDataset'


class RLVDataLoader(BaseDataset):
    def initialize(self, args, task):
        self.args = args
        self.low_img_dir = args.lowlight_images_path
        self.task = task
        self.train_low_data_names = []
        self.train_target_data_names = []
        assert os.path.exists(self.low_img_dir), "Input directory does not exist!"

        self.train_low_data_names = self.load_dataset_BVI(self.low_img_dir, task)

        self.count = len(self.train_low_data_names)
        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.last_data_name_path = self.train_low_data_names[0]

    def load_dataset_BVI(self, dir, task):
        img_list = []
        ll_10_dir_name = "low_light_10"
        ll_20_dir_name = "low_light_20"
        assert task == 'train' or task == 'test', "Invalid phase: " + str(task)

        phase_list_file = str(task) + '_list.txt'
        # image_list_lowlight = []
        with open(os.path.join(dir, phase_list_file), 'r') as file:
            phase_list = file.readlines()
            assert len(phase_list) > 0, "No input data."
        
        print("phase_list:", phase_list)
        
        for folder_name in phase_list:
            folder_name = folder_name.strip()
            print("folder_name:", folder_name)
            # train_ll_10_list = glob.glob(os.path.join(dir, 'input', folder_name, ll_10_dir_name, "*.png"))
            # train_ll_20_list = glob.glob(os.path.join(dir, 'input', folder_name, ll_20_dir_name, "*.png"))
            train_ll_10_list = glob.glob(os.path.join(dir, 'input', folder_name, ll_10_dir_name, "*.png"))
            train_ll_20_list = glob.glob(os.path.join(dir, 'input', folder_name, ll_20_dir_name, "*.png"))
            
            print("Path for low_light_10:", os.path.join(dir, 'input', folder_name, ll_10_dir_name, "*.png"))
            print("Path exists?", os.path.exists(os.path.join(dir, 'input', folder_name)))

            # sort file by name:
            train_ll_10_list = self.sort_files_by_name(train_ll_10_list)
            train_ll_20_list = self.sort_files_by_name(train_ll_20_list)

            img_list.extend(train_ll_10_list)
            img_list.extend(train_ll_20_list)
            print("train_ll_10_list ",train_ll_10_list)
            print("train_ll_20_list ",train_ll_20_list)
        print("img_list ",img_list)
        return img_list

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        new_size = (1920, 1080)
        im = im.resize(new_size)
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):
        ll = self.load_images_transform(self.train_low_data_names[index])
        img_name = os.path.splitext(os.path.basename(self.train_low_data_names[index]))[0]
        img_path = self.train_low_data_names[index]
        last_data_name_path = self.last_data_name_path
        self.last_data_name_path = img_path

        return ll, img_name, img_path, last_data_name_path

    def __len__(self):
        return self.count

    def name(self):
        return 'BVI-RLV'


class DidDataloader(BaseDataset):
    def initialize(self, args, task):
        self.args = args
        self.low_img_dir = args.lowlight_images_path
        self.task = task
        self.train_low_data_names = []
        self.train_target_data_names = []
        assert os.path.exists(self.low_img_dir), "Input directory does not exist!"

        self.train_low_data_names = self.load_dataset(self.low_img_dir, task)

        self.count = len(self.train_low_data_names)
        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.last_data_name_path = self.train_low_data_names[0]

    def load_dataset(self, dir, task):
        img_list = []

        assert task == 'train' or task == 'test', "Invalid phase: " + str(task)

        phase_list_file = str(task) + '_list.txt'
        # image_list_lowlight = []
        with open(os.path.join(dir, phase_list_file), 'r') as file:
            phase_list = file.readlines()
            assert len(phase_list) > 0, "No input data." 

        for folder_name in phase_list:
            folder_name = folder_name.strip()
            train_ll_list = glob.glob(os.path.join(dir, 'input', folder_name, "*.jpg"))
            train_ll_list.extend(glob.glob(os.path.join(dir, 'input', folder_name, "*.png")))

            # sort file by name:
            train_ll_list = self.sort_files_by_name(train_ll_list)

            img_list.extend(train_ll_list)

        return img_list

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        new_size = (1920, 1080)
        im = im.resize(new_size)
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):
        ll = self.load_images_transform(self.train_low_data_names[index])
        img_name = os.path.splitext(os.path.basename(self.train_low_data_names[index]))[0]
        img_path = self.train_low_data_names[index]
        last_data_name_path = self.last_data_name_path
        self.last_data_name_path = img_path

        return ll, img_name, img_path, last_data_name_path

    def __len__(self):
        return self.count

    def name(self):
        return 'DID'


class SDSDDataloader(BaseDataset):
    def initialize(self, args, task):
        self.args = args
        self.low_img_dir = args.lowlight_images_path
        self.task = task
        self.train_low_data_names = []
        self.train_target_data_names = []
        assert os.path.exists(self.low_img_dir), "Input directory does not exist!"

        # Auto-detect indoor/outdoor from directory structure
        indoor_dir = os.path.join(self.low_img_dir, 'indoor')
        outdoor_dir = os.path.join(self.low_img_dir, 'outdoor')
        
        # Load both indoor and outdoor datasets
        img_list = []
        if os.path.exists(indoor_dir):
            indoor_imgs = self.load_dataset_SDSD(self.low_img_dir, task, 'indoor')
            img_list.extend(indoor_imgs)
            print(f"Loaded {len(indoor_imgs)} indoor images")
            
        if os.path.exists(outdoor_dir):
            outdoor_imgs = self.load_dataset_SDSD(self.low_img_dir, task, 'outdoor')
            img_list.extend(outdoor_imgs)
            print(f"Loaded {len(outdoor_imgs)} outdoor images")
        
        self.train_low_data_names = img_list
        self.count = len(self.train_low_data_names)
        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.last_data_name_path = self.train_low_data_names[0] if self.count > 0 else ""

    def load_dataset_SDSD(self, dir, task, subset):
        img_list = []
        
        assert task == 'train' or task == 'test', "Invalid phase: " + str(task)
        assert subset in ['indoor', 'outdoor'], f"Invalid subset: {subset}. Must be 'indoor' or 'outdoor'"

        # SDSD has different file naming: sdsd_in_train.txt, sdsd_in_test.txt, sdsd_out_train.txt, sdsd_out_test.txt
        subset_prefix = 'in' if subset == 'indoor' else 'out'
        phase_list_file = f'sdsd_{subset_prefix}_{task}.txt'
        
        phase_list_path = os.path.join(dir, phase_list_file)
        print(f"Looking for SDSD phase list file: {phase_list_path}")
        
        if not os.path.exists(phase_list_path):
            print(f"Phase list file not found: {phase_list_path}")
            return img_list
            
        with open(phase_list_path, 'r') as file:
            phase_list = file.readlines()
            if len(phase_list) == 0:
                print("No input data in phase list.")
                return img_list
        
        print(f"SDSD phase_list for {subset}:", phase_list[:5])  # Print first 5 entries
        
        # SDSD structure: 3_SDSD/indoor/indoor_png/ or 3_SDSD/outdoor/outdoor_png/
        subset_png_dir_name = f"{subset}_png"
        subset_dir = os.path.join(dir, subset, subset_png_dir_name)
        print(f"Looking in subset directory: {subset_dir}")
        
        if not os.path.exists(subset_dir):
            print(f"Subset directory not found: {subset_dir}")
            return img_list
        
        for line in phase_list:
            pair_dir_name = line.strip()
            if not pair_dir_name:
                continue
            # Construct path to the pair directory
            current_pair_dir = os.path.join(subset_dir, pair_dir_name)
            
            if os.path.isdir(current_pair_dir):
                # Find image files in the directory
                img_files = glob.glob(os.path.join(current_pair_dir, '*.png'))
                img_files.extend(glob.glob(os.path.join(current_pair_dir, '*.jpg')))
                
                # Identify the low-light image (assuming it's not the ground truth)
                low_light_file = None
                for f in img_files:
                    if 'gt' not in f.lower() and 'normal' not in f.lower():
                        low_light_file = f
                        break  # Take the first non-GT image
                
                
                if low_light_file:
                    img_list.append(low_light_file)
                # If no specific file found, take the first image available
                elif len(img_files) > 0:
                    img_list.append(img_files[0])
                else:
                    print(f"No images found in directory: {current_pair_dir}")
            else:
                print(f"Could not find file or directory: {current_pair_dir} (from line '{pair_dir_name}')")

        # Sort file by name
        img_list = self.sort_files_by_name(img_list)
        print(f"Found {len(img_list)} images for SDSD {subset} {task}")
        
        return img_list

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        new_size = (1920, 1080)
        im = im.resize(new_size)
        img_norm = self.transform(im)
        return img_norm

    def __getitem__(self, index):
        ll = self.load_images_transform(self.train_low_data_names[index])
        img_name = os.path.splitext(os.path.basename(self.train_low_data_names[index]))[0]
        img_path = self.train_low_data_names[index]
        last_data_name_path = self.last_data_name_path
        self.last_data_name_path = img_path

        return ll, img_name, img_path, last_data_name_path

    def __len__(self):
        return self.count

    def name(self):
        return 'SDSD'