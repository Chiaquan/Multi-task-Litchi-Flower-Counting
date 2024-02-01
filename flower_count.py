import torch.nn as nn
import network
from Model import FlowerNet


class FlowerCounter(nn.Module):
    def __init__(self):
        super(FlowerCounter, self).__init__()
        self.DME = FlowerNet()
        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self,  im_data, gt_data=None):        
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)                
        density_map = self.DME(im_data)
        
        if self.training:
            gt_data_density = network.np_to_variable(gt_data['gt_density'], is_cuda=True, is_training=self.training)
            gt_data_mask = network.np_to_variable(gt_data['mask'], is_cuda=True,
                                                           is_training=self.training)
            gt = {'gt_density': gt_data_density, 'mask': gt_data_mask}
            self.loss_mse = self.build_loss(density_map, gt)
            
        return density_map
    
    def build_loss(self, density_map, gt_data, a=8):
        loss1 = self.loss_fn1(density_map['density'], gt_data['gt_density'])
        loss2 = self.loss_fn2(density_map['mask'], gt_data['mask'])
        loss = a * loss1 + loss2

        return loss
