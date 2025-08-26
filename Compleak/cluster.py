import torch
import torch.nn as nn
from cluster_dp import share_weight
import argparse
import copy
import os
import pickle
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from dataset import get_dataset
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)

parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--early_stop', default=10, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.0001, type=float)

parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="sgd", type=str)
parser.add_argument('--cluster_name', default='knn', type=str)
parser.add_argument('--cluster_num', default=16, type=int)

parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--t_max', default=50, type=int)

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.benchmark = True

    cluster_lr = args.lr
    save_folder_original = f"results/{args.dataset_name}_{args.model_name}"
    save_folder_cluster = f"results_compress/{args.dataset_name}_{args.model_name}_{args.cluster_name}_{args.cluster_num}"
    print(f"Save Folder: {save_folder_cluster}")

    
    trainset, testset = get_dataset(args.dataset_name)  
    total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder_original}/data_index.pkl"

    save_folder_cluster_victim= f"{save_folder_cluster}/victim_model"
    if not os.path.exists(save_folder_cluster_victim):
        os.makedirs(save_folder_cluster_victim)

    with open(data_path, 'rb') as f:
        victim_train_list, victim_test_list, attack_split_list = pickle.load(f)

    victim_train_dataset = Subset(total_dataset, victim_train_list)
    victim_test_dataset = Subset(total_dataset, victim_test_list)
    
    
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    victim_model_save_folder = save_folder_original + "/victim_model"
    victim_model_path = f"{victim_model_save_folder}/best.pth"

    victim_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path)
    test_acc, test_loss = victim_model.test(victim_test_loader, "Pretrained Victim")
    org_state = copy.deepcopy(victim_model.model.state_dict())
    

    victim_cluster_model = BaseModel(
        args.dataset_name, args.model_name, t_max = args.t_max, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=save_folder_cluster_victim,
        device=device, optimizer=args.optimizer, lr=cluster_lr, weight_decay=args.weight_decay)
    
    
    victim_cluster_model.model.load_state_dict(org_state)
    victim_cluster_model.model = share_weight(victim_cluster_model.model)

    print("before fine-tuning")
    test_acc, test_loss = victim_cluster_model.test(victim_test_loader, f"Test")
    print("fine-tuning")
    
    best_acc = 0
    count = 0
    for epoch in range(args.epochs):
        train_acc, train_loss = victim_cluster_model.cluster_train(epoch, victim_train_loader)
        test_acc, test_loss = victim_cluster_model.test(victim_test_loader, f"Epoch {epoch} Test")
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = victim_cluster_model.save(epoch, test_acc, test_loss)
            best_path = save_path

            count = 0
        elif args.early_stop > 0:
            count += 1
            if count > args.early_stop:
                print(f"Early Stop at Epoch {epoch}")
                break
    
    
    for shadow_ind in range(args.shadow_num):
        attack_train_list, attack_test_list = attack_split_list[shadow_ind]
        attack_train_dataset = Subset(total_dataset, attack_train_list)
        attack_test_dataset = Subset(total_dataset, attack_test_list)

        attack_train_loader = DataLoader(attack_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        attack_test_loader = DataLoader(attack_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        
        shadow_model_path = f"{save_folder_original}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(shadow_model_path)
        test_acc, _ = shadow_model.test(attack_test_loader, f"Pretrain Shadow")
        org_state = copy.deepcopy(shadow_model.model.state_dict())

        cluster_shadow_model_save_folder = f"{save_folder_cluster}/shadow_model_{shadow_ind}"
        if not os.path.exists(cluster_shadow_model_save_folder):
            os.makedirs(cluster_shadow_model_save_folder)

        shadow_cluster_model = BaseModel(args.dataset_name, args.model_name, t_max = args.t_max, 
                                 num_cls=args.num_cls, input_dim=args.input_dim, save_folder=cluster_shadow_model_save_folder, device=device, optimizer=args.optimizer, lr=cluster_lr, weight_decay=args.weight_decay)
        
        shadow_cluster_model.model.load_state_dict(org_state)
        shadow_cluster_model.model = share_weight(shadow_cluster_model.model)
    
        print("before fine-tuning")
        test_acc, test_loss = shadow_cluster_model.test(attack_test_loader, f"Test")
        print("fine-tuning")

        best_acc = 0
        count = 0

        for epoch in range(args.epochs):
            train_acc, train_loss = shadow_cluster_model.cluster_train(epoch, attack_train_loader)
            test_acc, test_loss = shadow_cluster_model.test(attack_test_loader, f"Epoch {epoch} Test")
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = shadow_cluster_model.save(epoch, test_acc, test_loss)
                best_path = save_path

                count = 0
            elif args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)



