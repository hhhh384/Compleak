import argparse
import json
import numpy as np
import os
import pickle
import shutil
import random
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from dataset import get_dataset,get_imagenet
from utils import seed_worker

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--dataset_name', default="cifar10", type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=30, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--early_stop', default=15, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="sgd", type=str)
parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--type', default=True, type=bool)
parser.add_argument('--warm', default=0, type=int)
parser.add_argument('--t_max', default=50, type=int)

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.benchmark = True

    save_folder = f"results/{args.dataset_name}_{args.model_name}"
    print(f"Save Folder: {save_folder}")
    

    trainset, testset = get_dataset(args.dataset_name)  
    total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder}/data_index.pkl"
    

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with open(data_path, 'rb') as f:
        victim_train_list, victim_test_list, attack_split_list = pickle.load(f)
 
    victim_train_dataset = Subset(total_dataset, victim_train_list)
    victim_test_dataset = Subset(total_dataset, victim_test_list)

    print(f"Total Data Size: {total_size}, "
          f"Victim Train Size: {len(victim_train_list)}, "
          f"Victim Test Size: {len(victim_test_list)}")
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    victim_model_save_folder = save_folder + "/victim_model"

    print("Train Victim Model")
    if not os.path.exists(victim_model_save_folder):
        os.makedirs(victim_model_save_folder)

    if args.warm:
        iter_per_epoch = len(victim_train_loader)
    else:
        iter_per_epoch = 0
    victim_model = BaseModel(
        args.dataset_name, args.model_name, t_max = args.t_max, warm=args.warm, iter_per_epoch=iter_per_epoch, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=victim_model_save_folder,
        device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    

    best_acc = 0
    count = 0
    for epoch in range(args.epochs):
        train_acc, train_loss = victim_model.train(epoch, victim_train_loader, f"Epoch {epoch} Train")
        test_acc, test_loss = victim_model.test(victim_test_loader, f"Epoch {epoch} Test")
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = victim_model.save(epoch, test_acc, test_loss)
            best_path = save_path
            count = 0
        elif args.early_stop > 0:
            count += 1
            if count > args.early_stop:
                print(f"Early Stop at Epoch {epoch}")
                break
    shutil.copyfile(best_path, f"{victim_model_save_folder}/best.pth")

    

    # Train shadow models
    for shadow_ind in range(args.shadow_num):
        attack_train_list, attack_test_list = attack_split_list[shadow_ind]
        print(f"Shadow Train Size: {len(attack_train_list)}, "
          f"Shadow Test Size: {len(attack_test_list)}")
        attack_train_dataset = Subset(total_dataset, attack_train_list)
        attack_test_dataset = Subset(total_dataset, attack_test_list)
        attack_train_loader = DataLoader(attack_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        attack_test_loader = DataLoader(attack_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        print(f"Train Shadow Model {shadow_ind}")
        shadow_model_save_folder = f"{save_folder}/shadow_model_{shadow_ind}"
        if not os.path.exists(shadow_model_save_folder):
            os.makedirs(shadow_model_save_folder)
        
        if args.warm:
            iter_per_epoch = len(attack_train_loader)
        else:
            iter_per_epoch = 0

        shadow_model = BaseModel(args.dataset_name, args.model_name, t_max = args.t_max, warm=args.warm, iter_per_epoch=iter_per_epoch, 
                                 num_cls=args.num_cls, input_dim=args.input_dim, save_folder=shadow_model_save_folder, device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            train_acc, train_loss = shadow_model.train(epoch, attack_train_loader, f"Epoch {epoch} Shadow Train")
            test_acc, test_loss = shadow_model.test(attack_test_loader, f"Epoch {epoch} Shadow Test")
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = shadow_model.save(epoch, test_acc, test_loss)
                best_path = save_path
                count = 0
            elif args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break

        shutil.copyfile(best_path, f"{shadow_model_save_folder}/best.pth")


if __name__ == '__main__':
    args = parser.parse_args()

    print(args)
    main(args)