import numpy as np
import torch
import random
from torch.nn import init
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            init.xavier_normal_(m.weight)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()


def get_model(model_type, num_cls, input_dim):
    if model_type == "resnet18":
        from net.resnet import resnet18
        model = resnet18(pretrained=False, num_classes=num_cls)
    elif model_type == "columnfc":
        from models import ColumnFC
        model = ColumnFC(input_dim=input_dim, output_dim=num_cls)
    elif model_type == 'resnet50':
        from net.resnet import resnet50
        model = resnet50()
    elif model_type == 'vgg16':
        from  net.vgg import vgg16_bn
        model = vgg16_bn(pretrained=False, num_classes=num_cls)
    elif model_type == "mobilenetv2":
        from net.mobilenetv2 import MobileNetV2
        model = MobileNetV2()
    else:
        print(model_type)
        raise ValueError
    return model


def get_optimizer(optimizer_name, parameters, lr, weight_decay=5e-4):
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer_name == "":
        optimizer = None
    else:
        print(optimizer_name)
        raise ValueError
    return optimizer

def get_train_transformer(name):
    if name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    elif name == "cifar100":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        ])

    else:
        raise ValueError

    return transform_train


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
