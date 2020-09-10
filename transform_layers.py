from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import utils
import torch.utils.cpp_extension
import random
import os

class FlipHorizontal(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_.flip([1])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class FlipVertical(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_.flip([0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Invert(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                ones = torch.ones(d_.size(), dtype=d_.dtype, layout=d_.layout, device=d_.device)
                tf = ones - d_
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class BinaryThreshold(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], float) or params[0] < -1 or params[0] > 1):
            print('Binary threshold parameter should be a float between -1 and 1.')
            # raise ValueError

        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                t = Variable(torch.Tensor([params[0]]))
                t = t.to(d_.device)
                tf = (d_ > t).float() * 1
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class ScalarMultiply(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isinstance(params[0], float)):
            print('Scalar multiply parameter should be a float')
            # raise ValueError

        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_ * params[0]
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Ablate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_ * 0
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class ManipulationLayer(nn.Module):
    def __init__(self, layerID):
        super().__init__()
        self.layerID = layerID
        
        # layers
        self.flip_h = FlipHorizontal()
        self.flip_v = FlipVertical()
        self.invert = Invert()
        self.binary_thresh = BinaryThreshold()
        self.scalar_multiply = ScalarMultiply()
        self.ablate = Ablate()
        
        self.layer_options = {
            "flip-h": self.flip_h,
            "flip-v": self.flip_v,
            "invert": self.invert,
            "binary-thresh": self.binary_thresh,
            "scalar-multiply": self.scalar_multiply,
            "ablate": self.ablate
        }

    def forward(self, input, tranforms_dict_list):
        out = input
        for transform_dict in tranforms_dict_list:
            if transform_dict['layerID'] == self.layerID:
                out = self.layer_options[transform_dict['transformID']](out, transform_dict['params'], transform_dict['indicies'])
        return out
    