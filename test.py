import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import os

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

# 修复：图像级评估（仅 sample AUROC，无像素级因为无 GT 掩码）
# 移除 gt_list_px, pr_list_px, aupro_list 和 compute_pro 调用
# gt_list_sp 使用 label (0/1)，而非 np.max(gt)（因为 gt 全 0）
def evaluation(encoder, bn, decoder, dataloader, device, _class_=None):
    bn.eval()
    decoder.eval()
    gt_list_sp = []
    pr_list_sp = []  # 使用 max anomaly score for sample-level
    with torch.no_grad():
        for img, gt, label, _ in dataloader:  # 正确解包
            img = img.to(device)
            if img.shape[1] == 1:  # 处理灰度图像
                img = img.repeat(1, 3, 1, 1)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # 跳过像素级：不处理 gt（全0），不计算 aupro
            gt_list_sp.extend(label.cpu().numpy())  # 使用 label 作为图像级 GT (0/1)
            pr_list_sp.append(np.max(anomaly_map))  # sample-level score: max anomaly

    auroc_px = 0.0  # 占位，无像素 GT
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    aupro_px = 0.0  # 占位，无像素 GT
    return auroc_px, auroc_sp, aupro_px

def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    # 使用 load_data 加载（一致性，替换硬编码路径）
    _, test_dataloader = load_data(dataset_name=_class_, batch_size=1, input_size=256)
    ckp_path = "/data/c425/cx/RD4AD/checkpoints/wres50_user3.pth"
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device, _class_)
    print(_class_, ':', auroc_px, ',', auroc_sp, ',', aupro_px)
    return auroc_px

import os

def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # 使用 load_data 加载（一致性）
    _, test_dataloader = load_data(dataset_name=_class_, batch_size=1, input_size=256)
    ckp_path = f'/data/c425/cx/RD4AD/checkpoints/wres50_user3.pth'
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:  # 正确解包
            if (label.item() == 0):
                continue
            #if count <= 10:
            #    count += 1
            #    continue

            decoder.eval()
            bn.eval()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            #inputs.append(feature)
            #inputs.append(outputs)
            #t_sne(inputs)

            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            # 修复：正确反归一化图像
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1) * 255
            img_np = img_np.astype(np.uint8)
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            #img = np.uint8(min_max_norm(img)*255)
            #if not os.path.exists('./results_all/'+_class_):
            #    os.makedirs('./results_all/'+_class_)
            #cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            #cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'ad.png', ano_map)
            plt.imshow(ano_map)
            plt.axis('off')
            #plt.savefig('ad.png')
            plt.show()

            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            count += 1
            #if count>20:
            #    return 0
                #assert 1==2


def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = '/data/c425/cx/RD4AD/checkpoints/wres50_user3.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)  # 假设 load_data 支持 name 和 _class_ 参数；否则调整

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    os.makedirs('./nd_results', exist_ok=True)
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:  # 修复解包
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            #if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            #print(inputs[-1].shape)
            outputs = decoder(bn(inputs))


            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            #anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            # 修复：正确反归一化图像
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1) * 255
            img_np = img_np.astype(np.uint8)
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            #img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            #img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
            #plt.imshow(ano_map)
            #plt.axis('off')
            #plt.savefig('ad.png')
            #plt.show()

            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            #count += 1
            #if count>40:
            #    return 0
                #assert 1==2
            gt_list_sp.extend(label.cpu().numpy())  # 使用 label (0/1)
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        # 移除 indx1/indx2（label 已为 0/1，无需转换）
        gt_list_sp = np.array(gt_list_sp)

        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        #print(type(vis_data))
        #np.save('vis.npy',vis_data)
        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)


# 注释 compute_pro（不再使用）
# def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
#     ...
#     assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"

def detection(encoder, bn, decoder, dataloader, device, _class_):
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, gt, label, _ in dataloader:  # 修复解包
            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            # label = label.to(device)  # 不需要
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.extend(label.cpu().numpy())  # 使用 label (0/1)
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        # 移除 indx1/indx2（label 已为 0/1）
        gt_list_sp = np.array(gt_list_sp)

        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean

if __name__ == '__main__':
    # 你要测试的类别名
    _class_ = 'user3'

    print('====== Running test() ======')
    test(_class_)

    # 如果你还想顺便看可视化，就再开下面这一行：
    # print('====== Running visualization() ======')
    # visualization(_class_)
