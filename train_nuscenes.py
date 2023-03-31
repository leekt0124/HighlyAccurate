#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os

import torchvision.utils

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataLoader.nuscenes_dataset import load_train_data, load_val_data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

import numpy as np
import os
import argparse

from utils import gps2distance
import time

# from models_kitti_360 import LM_G2SP, loss_func, LM_S2GP
from models_nuscenes import LM_G2SP, loss_func, LM_S2GP

# garbage collector
import gc

########################### ranking test ############################
def localize(net_localize, args,  device, save_path, best_rank_result, epoch):
    print("Runing localization pipeline")
    LOCALIZE_FILE = ""
    net_localize.eval()
    dataloader = load_localize_data( args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    
    start_time = time.time()
    for i, data in enumerate(dataloader):
        # If batch_size is not set to one, will return a list of tensor where the first dimension of the tensors are the batch_size
        # print("len(data) = ", len(data)) # 7
        # print("data[0].shape = ", data[0].shape) # satmap: torch.Size([B, 3, 512, 512])
        # print("data[1].shape = ", data[1].shape) # left_camera_k: torch.Size([B, 3, 3])
        # print("data[2].shape = ", data[2].shape) # groudn_left_img: torch.Size([B, 3, 256, 1024])
        sat_map, left_camera_k, grd_left_imgs = [item.to(device) for item in data]
        print("left_cam_k = ", left_camera_k)
        
        if args.direction == 'S2GP':
            shifts_lat, shifts_lon, theta = net_localize(sat_map, grd_left_imgs, mode='test')
            # print("shifts_lat = ", shifts_lat, ", shifts_lon = ", shifts_lon, "theta = ", theta)
        elif args.direction == 'G2SP':
            shifts_lat, shifts_lon, theta = net_localize(sat_map, grd_left_imgs, left_camera_k, mode='test')


def test1( net_test, args, save_path, best_rank_result, epoch, device):
    ### net evaluation state
    net_test.eval()
    dataloader = load_val_data(args.GrdImg_H, args.GrdImg_W, args.version, args.dataset_dir, args.labels_dir, args.loader, \
                               args.shift_range_lat, args.shift_range_lon, args.rotation_range, args.root_dir, args.zoom_level)
    pred_shifts = []
    pred_headings = []
    gt_shifts = []
    gt_headings = []

    start_time = time.time()
    for i, data in enumerate(dataloader, 0):

        if i > 10:
            break

        sat_map, grd_imgs, intrinsics, extrinsics, gt_shift_u, gt_shift_v, gt_heading, meter_per_pixel = [item.to(device) for item in data[:-1]]
        sample_name = data[-1]

        if args.direction == 'S2GP':   
            shifts_lat, shifts_lon, theta = net_test(sat_map, grd_imgs, intrinsics, extrinsics,  meter_per_pixel, sample_name, mode='test')     
            # shifts_lat, shifts_lon, theta = net_test(sat_map, grd_imgs, mode='test')
        elif args.direction == 'G2SP':
            shifts_lat, shifts_lon, theta = net_test(sat_map, grd_imgs, intrinsics, meter_per_pixel, mode='test')

        shifts = torch.stack([shifts_lat, shifts_lon], dim=-1)
        headings = theta.unsqueeze(dim=-1)
        gt_shift = torch.cat([gt_shift_v, gt_shift_u], dim=-1)  # [B, 2]

        if args.shift_range_lat==0 and args.shift_range_lon==0:
            loss = torch.mean(headings - gt_heading)
        else:
            loss = torch.mean(shifts_lat - gt_shift_u)
        loss.backward()  # just to release graph

        pred_shifts.append(shifts.data.cpu().numpy())
        pred_headings.append(headings.data.cpu().numpy())
        gt_shifts.append(gt_shift.data.cpu().numpy())
        gt_headings.append(gt_heading.data.cpu().numpy())        

        if i % 20 == 0:
            print(i)

    end_time = time.time()
    duration = (end_time - start_time)/len(dataloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    pred_headings = np.concatenate(pred_headings, axis=0) * args.rotation_range
    gt_shifts = np.concatenate(gt_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    gt_headings = np.concatenate(gt_headings, axis=0) * args.rotation_range
       
    print(f'gt_shifts.shape {gt_shifts.shape}')     # (200, 2)
    print(f'pred_shifts.shape {pred_shifts.shape}') # (800, 2)

    scio.savemat(os.path.join(save_path, 'Test1_results.mat'), {'gt_shifts': gt_shifts, 'gt_headings': gt_headings,
                                                         'pred_shifts': pred_shifts, 'pred_headings': pred_headings})

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))  # [N]
    angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]
    # angle_diff = angle_diff.numpy()

    init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
    init_angle = np.abs(gt_headings)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'Test1_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Time per image (second): ' + str(duration) + '\n')
    print('Validation results:')
    print('Init distance average: ', np.mean(init_dis))
    print('Pred distance average: ', np.mean(distance))
    print('Init angle average: ', np.mean(init_angle))
    print('Pred angle average: ', np.mean(angle_diff))


    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    diff_shifts = np.abs(pred_shifts - gt_shifts)
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

        pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum((angle_diff[:, 0] < angles[idx]) & (diff_shifts[:, 0] < metrics[idx])) / angle_diff.shape[0] * 100
        init = np.sum((init_angle[:, 0] < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])) / angle_diff.shape[0] * 100
        line = 'lat within ' + str(metrics[idx]) + ' & angle within ' + str(angles[idx]) + \
               ' (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.sum((distance < metrics[0]) & (angle_diff < angles[0])) / distance.shape[0] * 100

    net_test.train()

    ### save the best params
    if (result > best_rank_result):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(net_test.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    return result


def test2(net_test, args, save_path, best_rank_result, epoch,  device):
    ### net evaluation state
    net_test.eval()

    dataloader = load_test2_data(args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    pred_shifts = []
    pred_headings = []
    gt_shifts = []
    gt_headings = []

    start_time = time.time()

    for i, data in enumerate(dataloader, 0):
        sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in data[:-1]]

        # shifts_lat, shifts_lon, theta = net_test(sat_map, grd_left_imgs, mode='test')
        if args.direction == 'S2GP':
            shifts_lat, shifts_lon, theta = net_test(sat_map, grd_left_imgs, mode='test', level_first=args.level_first)
        elif args.direction == 'G2SP':
            shifts_lat, shifts_lon, theta = net_test(sat_map, grd_left_imgs, left_camera_k, mode='test')

        shifts = torch.stack([shifts_lat, shifts_lon], dim=-1)
        headings = theta.unsqueeze(dim=-1)
        # shifts: [B, 2]
        # headings: [B, 1]

        gt_shift = torch.cat([gt_shift_v, gt_shift_u], dim=-1)  # [B, 2]

        if args.shift_range_lat==0 and args.shift_range_lon==0:
            loss = torch.mean(headings - gt_heading)
        else:
            loss = torch.mean(shifts_lat - gt_shift_u)
        loss.backward()  # just to release graph

        pred_shifts.append(shifts.data.cpu().numpy())
        pred_headings.append(headings.data.cpu().numpy())
        gt_shifts.append(gt_shift.data.cpu().numpy())
        gt_headings.append(gt_heading.data.cpu().numpy())

        if i % 20 == 0:
            print(i)

    end_time = time.time()
    duration = (end_time - start_time)/len(dataloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    pred_headings = np.concatenate(pred_headings, axis=0) * args.rotation_range
    gt_shifts = np.concatenate(gt_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    gt_headings = np.concatenate(gt_headings, axis=0) * args.rotation_range

    scio.savemat(os.path.join(save_path, 'Test2_results.mat'), {'gt_shifts': gt_shifts, 'gt_headings': gt_headings,
                                                         'pred_shifts': pred_shifts, 'pred_headings': pred_headings})

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))  # [N]
    angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]
    # angle_diff = angle_diff.numpy()

    init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
    init_angle = np.abs(gt_headings)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, 'Test2_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Time per image (second): ' + str(duration) + '\n')
    print('Test results:')
    print('Init distance average: ', np.mean(init_dis))
    print('Pred distance average: ', np.mean(distance))
    print('Init angle average: ', np.mean(init_angle))
    print('Pred angle average: ', np.mean(angle_diff))


    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    diff_shifts = np.abs(pred_shifts - gt_shifts)
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

        pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum((angle_diff[:, 0] < angles[idx]) & (diff_shifts[:, 0] < metrics[idx])) / angle_diff.shape[0] * 100
        init = np.sum((init_angle[:, 0] < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])) / angle_diff.shape[0] * 100
        line = 'lat within ' + str(metrics[idx]) + ' & angle within ' + str(angles[idx]) + \
               ' (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    # result = np.sum((distance < metrics[0]) & (angle_diff < angles[0])) / distance.shape[0] * 100

    net_test.train()

    # ### save the best params
    # if (result > best_rank_result):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     torch.save(net_test.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    return


###### learning criterion assignment #######
def train(net, lr, args, device, save_path, model_save_path):
    bestRankResult = 0.0  # current best, Siam-FCANET18
    # loop over the dataset multiple times
    print(f'resume: {args.resume}')
    print(f'epochs: {args.epochs}')
    for epoch in range(args.resume, args.epochs):
        net.train()

        base_lr = lr
        base_lr = base_lr * ((1.0 - float(epoch) / 100.0) ** (1.0))

        print(base_lr)

        optimizer = optim.Adam(net.parameters(), lr=base_lr)
        # print("Model's state_dict:")
        # for param_tensor in net.state_dict():
            # print(param_tensor, "\t", net.state_dict()[param_tensor].size())

        optimizer.zero_grad()

        ### feeding A and P into train loader     
        trainloader = load_train_data(args.GrdImg_H, args.GrdImg_W, args.version, args.dataset_dir, args.labels_dir,  args.loader, \
                                      args.shift_range_lat, args.shift_range_lon, args.rotation_range, args.root_dir, args.zoom_level)

        loss_vec = []

        # For v1.0-trainval: batch_size:  4 => num of batches: 7033
        print(f'batch_size:  {args.loader.batch_size}\nnum of batches: {len(trainloader)}')


        # Initialize last_model_save_path
        last_model_save_path = model_save_path

        for Loop, Data in enumerate(trainloader, 0):
            # print("len(Data) = ", len(Data))
            # print("Loop = ", Loop)

            # if Loop > 5:
            #     import sys
            #     sys.exit()

            # Early stopping (leekt)
            # if Loop > 10:
            #     break
            # else:
            #     print("loop = ", Loop)

            # print(f'device: {device}')
            # get the inputs
            sat_map, grd_imgs, intrinsics, extrinsics, gt_shift_u, gt_shift_v, gt_heading, meter_per_pixel = [item.to(device) for item in Data[:-1]]
            # gt_shift_u.shape: (1, 1) (batch_size, )
            sample_name = Data[-1]
            print(f'sample_name: {sample_name}')
            # zero the parameter gradients
            optimizer.zero_grad()
     
            if args.direction == 'S2GP':
                # Check devices for sat_map and grd_imgs
                loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list = \
                    net(sat_map, grd_imgs, intrinsics, extrinsics, gt_shift_u, gt_shift_v, gt_heading, meter_per_pixel, sample_name, mode='train',
                        loop=Loop, level_first=args.level_first)
            elif args.direction =='G2SP':
                loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list = \
                    net(sat_map, grd_imgs, intrinsics, extrinsics, gt_shift_u, gt_shift_v, gt_heading, meter_per_pixel, sample_name, mode='train',)

            # print("loss = ", loss)
            # print("loss_drcrease = ", loss_decrease)
            loss.backward()

            optimizer.step()  # This step is responsible for updating weights
            optimizer.zero_grad()

            ### record the loss
            loss_vec.append(loss.item())

            if Loop % 10 == 9:  #
                level = abs(args.level) - 1
                # for level in range(len(shifts_decrease)):
                # print(loss_decrease[level].shape)
                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Delta: Level-' + str(level) +
                      ' loss: ' + str(np.round(loss.item(), decimals=4)) +
                      ' lat: ' + str(np.round(shift_lat_decrease[level].item(), decimals=2)) +
                      ' lon: ' + str(np.round(shift_lon_decrease[level].item(), decimals=2)) +
                      ' rot: ' + str(np.round(thetas_decrease[level].item(), decimals=2)))

                # print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Delta: Level-' + str(level) +
                #       ' loss: ' + str(np.round(loss_decrease[level].item(), decimals=4)) +
                #       ' lat: ' + str(np.round(shift_lat_decrease[level].item(), decimals=2)) +
                #       ' lon: ' + str(np.round(shift_lon_decrease[level].item(), decimals=2)) +
                #       ' rot: ' + str(np.round(thetas_decrease[level].item(), decimals=2)))

                if args.loss_method == 3:
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Last: Level-' + str(level) +
                          ' loss: ' + str(np.round(loss_last[level].item(), decimals=4)) +
                          ' lat: ' + str(np.round(shift_lat_last[level].item(), decimals=2)) +
                          ' lon: ' + str(np.round(shift_lon_last[level].item(), decimals=2)) +
                          ' rot: ' + str(np.round(theta_last[level].item(), decimals=2)) +
                          ' L1: ' + str(np.round(torch.sum(L1_loss).item(), decimals=2)) +
                          ' L2: ' + str(np.round(torch.sum(L2_loss).item(), decimals=2)) +
                          ' L3: ' + str(np.round(torch.sum(L3_loss).item(), decimals=2)) +
                          ' L4: ' + str(np.round(torch.sum(L4_loss).item(), decimals=2)))
                elif args.loss_method == 1 or args.loss_method == 2:
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Last: Level-' + str(level) +
                          ' loss: ' + str(np.round(loss_last[level].item(), decimals=4)) +
                          ' lat: ' + str(np.round(shift_lat_last[level].item(), decimals=4)) +
                          ' lon: ' + str(np.round(shift_lon_last[level].item(), decimals=4)) +
                          ' rot: ' + str(np.round(theta_last[level].item(), decimals=4)) +
                          ' L1: ' + str(np.round(torch.sum(L1_loss).item(), decimals=2)))
                else:
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Last: Level-' + str(level) +
                          ' loss: ' + str(np.round(loss_last[level].item(), decimals=4)) +
                          ' lat: ' + str(np.round(shift_lat_last[level].item(), decimals=2)) +
                          ' lon: ' + str(np.round(shift_lon_last[level].item(), decimals=2)) +
                          ' rot: ' + str(np.round(theta_last[level].item(), decimals=2))
                          )

            # Save the model every 100 loop in a single epoch
            if Loop % 10 == 9:
                # print('save model ...')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                old_path = last_model_save_path

                model_save_path_splits = model_save_path.split('_')
                model_save_path_splits[-1] = "Loop" + str(Loop)
                new_model_save_path = os.path.join(model_save_path_splits[0] + '_' + \
                                                   model_save_path_splits[1] + '_' + \
                                                   model_save_path_splits[2] + '_' + \
                                                   model_save_path_splits[3] + '_' + \
                                                   model_save_path_splits[4] + '_' + \
                                                   model_save_path_splits[5] + '.pth')
                print(f'save model ... {new_model_save_path}')
                torch.save(net.state_dict(), new_model_save_path)
                if os.path.exists(old_path):
                    os.remove(old_path)

                last_model_save_path = new_model_save_path

        ### save modelget_similarity_fn
        compNum = epoch % 100
        print('taking snapshot ...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), model_save_path)
        # torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(compNum) + '.pth'))

        ### ranking test        
        current = test1(net, args, save_path, bestRankResult, epoch, device)
        if (current > bestRankResult):
            bestRankResult = current

        # test2(net, args, save_path, bestRankResult, epoch, device)

    print('Finished Training')



def getSavePath(args):
    '''
        Get the directory where you want to save the trained model.
        Current working directory: /home/goroyeh/Yujiao/leekt/HighlyAccurate/outputs/2023-03-28/18-26-44/
        ModelsNuscenes/ :          /home/goroyeh/Yujiao/leekt/HighlyAccurate/ModelsNuscenes/
    '''

    # if args.test and args.use_default_model:
    #     save_path = '/mnt/workspace/datasets/yujiao_data/Models/ModelsKitti/LM_S2GP/lat20.0m_lon20.0m_rot10.0_Lev3_Nit5_Wei0_Dam0_Load0_LM_loss0_100.0_100.0_100.0_100.0_100.0_100.0_100.0'
    # elif not args.test and args.use_default_model:
    #     raise Exception("Can not use default model in non-testing mode")
    # else:
    #     save_path = './ModelsKitti/LM_' + str(args.direction) \
    #             + '/lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
    #     args.rotation_range) \
    #             + '_Lev' + str(args.level) + '_Nit' + str(args.N_iters) \
    #             + '_Wei' + str(args.using_weight) \
    #             + '_Dam' + str(args.train_damping) \
    #             + '_Load' + str(args.Load) + '_' + str(args.Optimizer) \
    #             + '_loss' + str(args.loss_method) \
    #             + '_' + str(args.coe_shift_lat) + '_' + str(args.coe_shift_lon) + '_' + str(args.coe_heading) \
    #             + '_' + str(args.coe_L1) + '_' + str(args.coe_L2) + '_' + str(args.coe_L3) + '_' + str(args.coe_L4)

    # if args.level_first:
    #     save_path += '_Level1st'

    # if args.proj != 'geo':
    #     save_path += '_' + args.proj

    # if args.use_gt_depth:
    #     save_path += '_depth'

    # if args.use_hessian:
    #     save_path += '_Hess'

    # if args.dropout > 0:
    #     save_path += '_Dropout' + str(args.dropout)

    # if args.damping != 0.1:
    #     save_path += '_Damping' + str(args.damping)


    # TODO: Goro change this for convenience
    # save_path = '../../../ModelsKitti/'
    save_path = '../../../ModelsNuscenes/'
    # save_path = '.'
    print(f'save_path: {save_path}, CWD: {Path.cwd()}')

    return save_path


from pathlib import Path
CONFIG_PATH = Path.cwd() / 'transformer/config'
CONFIG_NAME = 'config.yaml'
import hydra

from datetime import datetime

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    save_path = getSavePath(cfg.highlyaccurate)
    print(f'save_path: {save_path}')

    net = eval('LM_' + cfg.highlyaccurate.direction)(cfg) # class LM_S2GP
    ### cudaargs.epochs, args.debug)
    gc.collect()

    torch.cuda.empty_cache()   
    net.to(device)
    ###########################

    if cfg.highlyaccurate.localize:
        net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')))
        localize(net, cfg.highlyaccurate, device, save_path, 0., epoch=0)
    else:
        if cfg.highlyaccurate.test:
            # net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')))
            test_model_path =  "/home/goroyeh/Yujiao/leekt/HighlyAccurate/ModelsNuscenes/model_0329|0|27v1.0-trainval_epoch5_batch4_lr0.0001_Loop7029.pth"
            net.load_state_dict(torch.load(test_model_path))
            # net.load_state_dict(torch.load(os.path.join(save_path, 'model_1.pth')))
            # test1(net, cfg, save_path, 0., epoch=0)
            test1(net, cfg.highlyaccurate, save_path, 0, epoch=0, device=device)
            # test2(net, cfg, save_path, 0., epoch=0)
        
        else:
            date           = str(cfg.highlyaccurate.date)  # e.g. 0328
            dataset_str        = str(cfg.data.version)
            cur_datetime = datetime.now()
            # date   = str(cur_datetime.month)+str(cur_datetime.day)
            hour   = str(cur_datetime.hour)
            minute = str(cur_datetime.minute)
            datetime_str = date + '|' + hour +'|'+ minute + dataset_str

            epoch_str      = str(cfg.highlyaccurate.epochs)
            batch_size_str = str(cfg.loader.batch_size)
            lr_str         = str(cfg.highlyaccurate.lr)

            model_path_name = os.path.join(save_path, 'model_' + datetime_str + '_epoch'+ epoch_str + \
                                           '_batch'+ batch_size_str +'_lr'+lr_str + '_Loop0.pth')

            print(f'model_path_name: {model_path_name}')
            if cfg.highlyaccurate.resume:
                net.load_state_dict(torch.load(model_path_name))
                print("resume from" + model_path_name)
                # net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(cfg.highlyaccurate.resume - 1) + '.pth')))
                # print("resume from " + 'model_' + str(cfg.highlyaccurate.resume - 1) + '.pth')

            if cfg.highlyaccurate.visualize:
                net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')))
                # net.load_state_dict(torch.load(os.path.join(save_path, 'model_1.pth')))

            lr = cfg.highlyaccurate.lr

            train(net, lr, cfg.highlyaccurate, device, save_path, model_path_name)


if __name__ == '__main__':
    main()
  