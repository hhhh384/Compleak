import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import csc_matrix, csr_matrix


class NetCodebook():
    def __init__(self, conv_bits, fc_bits):
        self.conv_bits = conv_bits
        self.fc_bits = fc_bits
        self.codebook_index = []
        self.codebook_value = []

    def add_layer_codebook(self, layer_codebook_index, layer_codebook_value):
        self.codebook_index.append(layer_codebook_index)
        self.codebook_value.append(layer_codebook_value)


def apply_weight_sharing(model, bits=5):
    for module in model.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2**bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev)



def load_model(net):
    conv_layer_num = 0
    fc_layer_num = 0
    nz_num = []
    conv_value_array = []
    fc_value_array = []
    layer_types = []
    
    for name, param in net.named_parameters():
        if 'conv' in name:
            conv_layer_num += 1
            conv_weight = param.data.numpy()
            non_zero_values = conv_weight[conv_weight != 0]    
            nz_num.append(len(non_zero_values))
            conv_value_array.extend(non_zero_values)
            layer_types.append('conv')
        elif 'fc' in name:
            fc_layer_num += 1
            fc_weight = param.data.numpy()
            non_zero_values = fc_weight[fc_weight != 0]
            nz_num.append(len(non_zero_values))
            fc_value_array.extend(non_zero_values)
            layer_types.append('fc')

    conv_value_array = np.array(conv_value_array, dtype=np.float32)
    fc_value_array = np.array(fc_value_array, dtype=np.float32)

    return conv_layer_num, fc_layer_num, nz_num, conv_value_array, fc_value_array, layer_types


def share_weight(net, bits=4):
    for module in net.modules():  
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Processing {module.__class__.__name__}")
            dev = module.weight.device  
            weight = module.weight.data.cpu().numpy()  
            min_ = np.min(weight)  
            max_ = np.max(weight)  
            n_clusters = 2 ** bits
            space = np.linspace(min_, max_, num=n_clusters, dtype=np.float32)
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, algorithm="lloyd")
            kmeans.fit(weight.reshape(-1, 1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(weight.shape) 
            module.weight.data = torch.from_numpy(new_weight).to(dev) 

    return net

    
