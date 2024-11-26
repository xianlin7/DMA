import math
import torchvision
import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.imgname import read_img_name
import seaborn as sns
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch.nn.functional as Fn

def visual_segmentation(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    # img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    # img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(opt.img_size, opt.img_size))
    img_ori0 = cv2.resize(img_ori0, dsize=(opt.img_size, opt.img_size))

    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                    [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135],
                    [237, 145, 33], [176, 224, 230], [61, 145, 64], [227, 207, 87], [189, 252, 201], [245, 222, 179]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
        # img_r[seg0 == i] = table[i + 1 - 1, 0]
        # img_g[seg0 == i] = table[i + 1 - 1, 1]
        # img_b[seg0 == i] = table[i + 1 - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    img = cv2.addWeighted(img_ori0, 0.6, overlay, 0.4, 0) # ACDC
    #img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) # ISIC
    #img = np.uint8(0.3 * overlay + 0.7 * img_ori)
          
    fulldir = opt.result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)


# ----------------------------- cluster visualizzation -------------------------------
def generate_token_color(tokencolor, n_classes, n_nprototypes0, n_nprototypes, n_pprototypes):
    #n_ctoken = tokencolor.shape[1]//n_classes
    n_c0token = n_nprototypes
    down_time = n_nprototypes0//n_nprototypes

    new_tokencolor = tokencolor*1 # [B N 3]
    start_gray = 20
    inter_gray = (250-start_gray)// n_c0token
    for i in range(n_c0token):
        new_tokencolor[:, i, 0] = (start_gray + i*inter_gray)/255
        new_tokencolor[:, i, 1] = (start_gray + i*inter_gray)/255
        new_tokencolor[:, i, 2] = (start_gray + i*inter_gray)/255
    table = torch.tensor([[134, 131, 0], [154, 150, 0], [168, 164, 0], [184, 180, 0], [200, 195, 0], [222, 217, 0], [234, 228, 0], [244, 238, 0],
                          [176, 134, 0], [210, 160, 0], [234, 178, 0], [254, 194, 0], [255, 209, 63], [255, 222, 117], [255, 234, 167], [255, 243, 205],
                          [230, 93, 0], [255, 106, 5], [255, 128, 41], [255, 143, 67], [255, 161, 97], [255, 178, 125], [255, 198, 159], [255, 216, 189],
                          [168, 84, 0], [204, 102, 0], [238, 119, 0], [255, 148, 41], [255, 165, 75], [255, 182, 109], [255, 192, 129], [255, 213, 171],
                          [176, 0, 0], [204, 0, 0], [230, 0, 0], [255, 0, 0], [255, 91, 91], [255, 147, 147], [255, 179, 179], [255, 201, 201],
                          [204, 0, 83], [230, 0, 93], [255, 25, 118], [255, 75, 148], [255, 109, 168], [255, 147, 191], [255, 167, 203], [255, 197, 220],
                          [102, 0, 51], [146, 0, 73], [180, 0, 90], [218, 0, 109], [250, 0, 125], [255, 67, 161], [255, 113, 184], [255, 147, 201],
                          [155, 37, 116], [186, 44, 139], [210, 66, 162], [219, 103, 180], [227, 137, 197], [233, 165, 210], [241, 197, 226], [245, 215, 235]])
    table = table[None, :, :].cuda()/255
    table = table.type_as(tokencolor)
    table = table.repeat(tokencolor.shape[0], 1, 1)
    down_time2 = n_nprototypes0//n_pprototypes
    for i in range(tokencolor.shape[1]-n_c0token):
        new_tokencolor[:, i+n_c0token, 0] = table[:, i*down_time2, 2]
        new_tokencolor[:, i+n_c0token, 1] = table[:, i*down_time2, 1]
        new_tokencolor[:, i+n_c0token, 2] = table[:, i*down_time2, 0]
    return new_tokencolor

from models.components.tcformer_parts import map2token, token2map
def vis_cluster_results(x, prototypes, idx2cluster, id_layer, n_classes, n_nprototypes0, n_nprototypes, n_pprototypes, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens
    Return:
        vis_img (Tensor[B, 3, H, W]): visualize result.
    """
    transf = transforms.ToTensor()
    imgpath = read_img_name()
    img0 = cv2.imread(imgpath, 1)
    img0 = cv2.resize(img0, (256, 256))
    img = F.to_tensor(img0)[None, :, :].cuda()

    color_map = Fn.avg_pool2d(img, kernel_size=8)
    B, C, H, W = color_map.shape

    n_sequence = x.shape[1]
    h, w = int(math.sqrt(n_sequence)), int(math.sqrt(n_sequence))
    n_tokens = prototypes.shape[1]
    idx_token = torch.arange(n_tokens)[None, :].repeat(B, 1) # [b 4096], i.e., [0 1 2 3 4 ... 4095]
    agg_weight = x.new_ones(B, n_sequence, 1) # [B 4096 1]=[1 1 1 .... 1]

    out_dict = {}
    out_dict['x'] = prototypes
    out_dict['token_num'] = n_tokens
    out_dict['map_size'] = [h, w],
    out_dict['init_grid_size'] = [H, W]
    out_dict['idx_token'] = idx2cluster[:, n_tokens:] # [B N]

    #idx_token = idx2cluster[:, n_tokens:]

    token_color = map2token(color_map, out_dict) # [B N C]
    #token_color = generate_token_color(token_color, n_classes, n_nprototypes0, n_nprototypes, n_pprototypes)

    tmp_dict = out_dict.copy()
    tmp_dict['map_size'] = [H, W]
    tmp_dict['x'] = token_color
    vis_img = token2map(tmp_dict)

    token_idx = torch.arange(n_tokens)[None, :, None].float().cuda() / n_tokens
    tmp_dict['x'] = token_idx
    idx_map = token2map(tmp_dict)  # [B, 1, H, W]

    vis_img = Fn.interpolate(vis_img, [H * 16, W * 16], mode='nearest')
    idx_map = Fn.interpolate(idx_map, [H * 16, W * 16], mode='nearest')

    kernel = idx_map.new_zeros([4, 1, 3, 3])
    kernel[:, :, 1, 1] = 1
    kernel[0, :, 0, 1] = -1
    kernel[1, :, 2, 1] = -1
    kernel[2, :, 1, 0] = -1
    kernel[3, :, 1, 2] = -1

    for i in range(edge_width):
        edge_map = Fn.conv2d(Fn.pad(idx_map, [1, 1, 1, 1], mode='replicate'), kernel)
        edge_map = (edge_map != 0).max(dim=1, keepdim=True)[0]
        idx_map = idx_map * (~edge_map) + torch.rand(idx_map.shape).cuda() * edge_map

    edge_color = torch.tensor(edge_color)[None, :, None, None].cuda()
    vis_img = vis_img * (~edge_map) + edge_color * edge_map #[B 3 H W]
    #print("depth:", id_layer, "min:", torch.min(vis_img), "max:", torch.max(vis_img))
    vis_img = vis_img[0, :, :, :].cpu()
    vis_img = vis_img.permute(1, 2, 0)

    vis_img = np.uint8(vis_img*255)
          
    fulldir = "./visualization/Merge/INSTANCE/" + "SETR" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    image_filename = imgpath.split('/')[-1]
    image_filename = image_filename[:-4]
    cv2.imwrite(fulldir + image_filename + '_' + str(id_layer) + '.png', vis_img)
    return vis_img

def vis_grid_results(x, prototypes, idx2cluster, id_layer, n_classes, n_cprototypes, k_class0, edge_color=[1.0, 1.0, 1.0], edge_width=1):
    """Visualize tokens
    Return:
        vis_img (Tensor[B, 3, H, W]): visualize result.
    """
    transf = transforms.ToTensor()
    imgpath = read_img_name()
    img0 = cv2.imread(imgpath, 1)
    img0 = cv2.resize(img0, (256, 256))
    img = F.to_tensor(img0)[None, :, :].cuda()

    
    vis_img = img
    vis_img[:, :, 0:256:8, :] =1
    vis_img[:, :, :, 0:256:8] =1
    vis_img = vis_img[0, :, :, :].cpu()
    vis_img = vis_img.permute(1, 2, 0)

    vis_img = np.uint8(vis_img*255)
          
    fulldir = "./visualization/Merge/" + "instanceP8" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    image_filename = imgpath.split('/')[-1]
    image_filename = image_filename[:-4]
    cv2.imwrite(fulldir + image_filename + '_' + str(id_layer) + '.png', vis_img)
    return vis_img
