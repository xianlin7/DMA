import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, SoftDiceLoss
from utils.config import get_config
from models.models import get_model
from utils.evaluation import get_eval
from thop import profile

def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SETR_DSFormer', type=str, help='type of model')
    parser.add_argument('--task', default='INSTANCE', help='task or dataset name')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter

    opt.mode = "eval"
    opt.modelname = args.modelname
    opt.visual = False
    print(opt.load_path)      
    print("dataset:" + args.task + " -----------model name: "+ args.modelname)

    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        from utils.utils_rgb import JointTransform2D, ImageToImage2D

    # torch.backends.cudnn.enabled = True # Whether to use nondeterministic algorithms to optimize operating efficiency
    # torch.backends.cudnn.benchmark = True

    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 300  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ============================================= model initialization ==============================================
    tf_test = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    test_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_test, opt.classes)  # return image, mask, and filename
    testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    model = get_model(modelname=args.modelname, img_size=opt.img_size, img_channel=opt.img_channel, classes=opt.classes)
    model.to(device)
    model.load_state_dict(torch.load(opt.load_path))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total_params: {}".format(pytorch_total_params))
    # input = torch.randn(1, 1, 256, 256).cuda()
    # flops, params = profile(model, inputs=(input,) )
    # print('Gflops:', flops/1000000000, 'params:', params)

    criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1)

    if opt.mode == "train":
        dices, mean_dice, _, val_losses = get_eval(testloader, model, criterion=criterion, opt=opt, args=args)
        print("mean dice:", mean_dice)
    else:
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(testloader, model, criterion=criterion, opt=opt, args=args)
        print(mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp)
        print(std_dice, std_hdis, std_iou, std_acc, std_se, std_sp)
    
if __name__ == '__main__':
    main()
            


