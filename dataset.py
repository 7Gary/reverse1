from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np


def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        self.phase = phase  # 存储 phase 用于条件加载
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = None  # 无地面真相掩码
        self.transform = transform
        self.gt_transform = gt_transform
        # 加载数据集
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        # 调试：打印目录内容
        print(f"Debug: Checking directory '{self.img_path}'")
        if not os.path.exists(self.img_path):
            raise ValueError(f"Directory '{self.img_path}' does not exist!")
        all_items = os.listdir(self.img_path)
        print(f"Debug: Found {len(all_items)} items in '{self.img_path}': {all_items[:10]}...")  # 前10个

        if self.phase == "train":
            # 训练：从 train/good/ 子文件夹加载（标准 MVTec）
            good_dir = os.path.join(self.img_path, 'good')
            if not os.path.exists(good_dir):
                raise ValueError(f"Directory '{good_dir}' does not exist! Check if images are in train/good/")
            good_files = os.listdir(good_dir)
            print(f"Debug: Found {len(good_files)} items in '{good_dir}': {good_files[:10]}...")
            for filename in sorted(good_files):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 支持常见格式
                    img_path = os.path.join(good_dir, filename)
                    img_tot_paths.append(img_path)
                    gt_tot_paths.append(0)  # 无掩码，使用 0
                    tot_labels.append(0)
                    tot_types.append('good')  # 训练全为 normal
            print(f"Debug: Matched {len(img_tot_paths)} image files in train/good/")
        else:
            # 测试：标准 MVTec 结构，子文件夹 'good' 和缺陷类型
            defect_types = [d for d in all_items if os.path.isdir(os.path.join(self.img_path, d))]
            print(f"Debug: Found {len(defect_types)} subdirs in test: {defect_types}")
            for defect_type in defect_types:
                subdir_path = os.path.join(self.img_path, defect_type)
                sub_files = os.listdir(subdir_path)
                matched_sub = []
                for filename in sorted(sub_files):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        matched_sub.append(os.path.join(subdir_path, filename))
                if defect_type == 'good':
                    img_tot_paths.extend(matched_sub)
                    gt_tot_paths.extend([0] * len(matched_sub))  # 无掩码，使用 0
                    tot_labels.extend([0] * len(matched_sub))
                    tot_types.extend(['good'] * len(matched_sub))
                else:  # 缺陷类型
                    img_tot_paths.extend(matched_sub)
                    gt_tot_paths.extend([0] * len(matched_sub))  # 无掩码，使用 0
                    tot_labels.extend([1] * len(matched_sub))
                    tot_types.extend([defect_type] * len(matched_sub))
                print(f"Debug: In subdir '{defect_type}': {len(matched_sub)} images")

        print(f"Loaded {len(img_tot_paths)} images for phase '{self.phase}'")
        if len(img_tot_paths) == 0:
            print(f"Debug: No images matched. Check extensions (jpg/JPG/png) or subdirs.")
            raise ValueError(f"No images found in {self.img_path}. Ensure files in train/good/ for train phase.")
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # 由于无地面真相掩码，始终使用零张量作为 gt
        h, w = img.size()[-2], img.size()[-1]
        gt = torch.zeros([1, h, w])

        return img, gt, label, img_type


def load_data(dataset_name='user3', batch_size=16, input_size=224):
    """
    修改后的 load_data，仅支持指定路径的自定义 MVTec-like 数据集。
    图像为 2048x1024，因此变换调整为正方形 (input_size)。
    假设无地面真相掩码 (gt 始终为零张量，用于图像级异常检测)。
    """
    root = '/data/c425/cx/Patchcore_net/datasets/mvtec_anomaly_detection/mvtec_anomaly_detection/user3/'

    if dataset_name == 'user3':
        # 使用更大的初始调整大小处理 2048x1024，然后裁剪为正方形
        size = input_size * 2  # 例如 448，更好处理宽高比
        isize = input_size  # 最终大小，例如 224
        data_transforms, gt_transforms = get_data_transforms(size, isize)

        train_dataset = MVTecDataset(root=root, transform=data_transforms, gt_transform=gt_transforms, phase='train')
        test_dataset = MVTecDataset(root=root, transform=data_transforms, gt_transform=gt_transforms, phase='test')

        print(f"MVTec User3 数据集加载完成...")
        print(f"训练样本: {len(train_dataset)} (全正常)")
        print(f"测试样本: {len(test_dataset)} (正常 + 异常)")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
        )

        return train_dataloader, test_dataloader
    else:
        raise Exception(
            "您输入的 dataset 为 {}，但仅支持 'user3'！".format(dataset_name))