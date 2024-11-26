# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation
import pandas as pd
import time

def obtain_patien_id(filename):
    if "-" in filename: # filename = "xx-xx-xx_xxx"
        filename = filename.split('-')[-1]
    # filename = xxxxxxx or filename = xx_xxx
    if "_" in filename:
        patientid = filename.split("_")[0]
    else:
        patientid = filename[:3]
    return patientid

def eval_2d_slice(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    for batch_idx, (input_image, ground_truth, mask_mini, *rest) in enumerate(valloader):
        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        if isinstance(rest[0][0], str):
            image_filename = rest[0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        test_img_path = os.path.join(opt.data_path + '/img', image_filename[0])
        from utils.imgname import keep_img_name
        keep_img_name(test_img_path)

        with torch.no_grad():
            start_time = time.time()
            predict = model(input_image)
            sum_time = time.time()-start_time
            #print(sum_time)

        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            for i in range(0, opt.classes):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, :, :] == i] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, :, :] == i] = 255
                dices[eval_number+j, i] += metrics.dice_coefficient(pred_i, gt_i)
                iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
                ious[eval_number+j, i] += iou
                accs[eval_number+j, i] += acc
                ses[eval_number+j, i] += se
                sps[eval_number+j, i] += sp
                hds[eval_number+j, i] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
                del pred_i, gt_i
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
        eval_number = eval_number + b
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    if opt.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def eval_2d_patient(valloader, model, criterion, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    patientnumber = 500  # maxnum patient number
    flag = np.zeros(patientnumber)  # record the patients
    tps, fps = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    tns, fns = np.zeros((patientnumber, opt.classes)), np.zeros((patientnumber, opt.classes))
    hds = np.zeros((patientnumber, opt.classes))
    for batch_idx, (input_image, ground_truth, mask_mini, *rest) in enumerate(valloader):
        if isinstance(rest[0][0], str):
            image_filename = rest[0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        test_img_path = os.path.join(opt.data_path + '/img', image_filename[0])
        from utils.imgname import keep_img_name
        #keep_img_name(test_img_path)

        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        with torch.no_grad():
            predict = model(input_image)
        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            patientid = int(obtain_patien_id(image_filename[j]))
            flag[patientid] += 1
            for i in range(1, opt.classes):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, :, :] == i] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, :, :] == i] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
                hd = hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
                if hd > hds[patientid, i]:
                    hds[patientid, i] = hd
                tps[patientid, i] += tp
                fps[patientid, i] += fp
                tns[patientid, i] += tn
                fns[patientid, i] += fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], image_filename[j], opt)
    tps = tps[flag > 0, :]
    fps = fps[flag > 0, :]
    tns = tns[flag > 0, :]
    fns = fns[flag > 0, :]
    hds = hds[flag > 0, :]
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    dices = np.mean(patient_dices, axis=0)  # c
    hdis = np.mean(hds, axis=0)
    val_losses = val_losses / (batch_idx + 1)
    mean_dice = np.mean(dices[1:], axis=0)
    mean_hdis = np.mean(hdis[1:], axis=0)
    if opt.mode=="train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean = np.mean(patient_dices*100, axis=0)
        dices_std = np.std(patient_dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth)*100 # p c
        iou_mean = np.mean(iou, axis=0) 
        iou_std = np.std(iou, axis=0)
        acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)*100
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        se = (tps + smooth) / (tps + fns + smooth)*100
        se_mean = np.mean(se, axis=0)
        se_std = np.std(se, axis=0)
        sp = (tns + smooth) / (fps + tns + smooth)*100
        sp_mean = np.mean(sp, axis=0)
        sp_std = np.std(sp, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std


def get_eval(valloader, model, criterion, opt, args=None):
    if opt.eval_mode == "slice":
        return eval_2d_slice(valloader, model, criterion, opt)
    elif opt.eval_mode == "patient":
        return eval_2d_patient(valloader, model, criterion, opt)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)