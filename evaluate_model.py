import torch
import time
from flower_count import FlowerCounter
import network
import numpy as np
import torch.nn as nn


def evaluate_model(trained_model, data_loader):
    net = FlowerCounter()
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0
    for blob in data_loader:                        
        im_data = blob['data']
        gt_data = blob
        density_map = net(im_data, gt_data)
        density_mapa = density_map['density'].data.cpu().numpy()
        mask = torch.sigmoid(density_map['mask']).data.cpu().numpy()
        result = density_mapa * mask
        gt_count = np.sum(gt_data['gt_density'])
        et_count = np.sum(result)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))

    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    return mae, mse