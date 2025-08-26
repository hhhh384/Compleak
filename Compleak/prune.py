import argparse
import copy
import json
import numpy as np
import os
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from dataset import get_dataset,get_imagenet
from pruner import get_pruner
from utils import seed_worker
torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--early_stop', default=10, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="sgd", type=str)
parser.add_argument('--prune_epochs', default=50, type=int)
parser.add_argument('--pruner_name', default='l1unstructure', type=str)
parser.add_argument('--prune_sparsity', default=0.8, type=float)
parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--t_max', default=50, type=int)

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    prune_lr = args.lr
    save_folder_original = f"results/{args.dataset_name}_{args.model_name}"
    save_folder_pruned = f"results_compress/{args.dataset_name}_{args.model_name}_{args.pruner_name}_{args.prune_sparsity}"

    print(f"Save Folder: {save_folder_pruned}")

    trainset, testset = get_dataset(args.dataset_name)
    total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder_original}/data_index.pkl"

    
    save_folder_pruned_victim= f"{save_folder_pruned}/victim_model"
    if not os.path.exists(save_folder_pruned_victim):
        os.makedirs(save_folder_pruned_victim)
    

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
    
    print("Prune Victim Model")
    org_state = copy.deepcopy(victim_model.model.state_dict())

    victim_pruned_model = BaseModel(
        args.dataset_name, args.model_name, t_max = args.t_max, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=save_folder_pruned_victim,
        device=device, optimizer=args.optimizer, lr=prune_lr, weight_decay=args.weight_decay)
    
    victim_pruned_model.model.load_state_dict(org_state)
    pruner = get_pruner(args.pruner_name, victim_pruned_model.model, sparsity=args.prune_sparsity)
    victim_pruned_model.model = pruner.compress()
   
    best_acc = 0
    count = 0
    for epoch in range(args.prune_epochs):
        pruner.update_epoch(epoch)
        train_acc, train_loss = victim_pruned_model.train(epoch, victim_train_loader, f"Epoch {epoch} Prune Train")
        test_acc, test_loss = victim_pruned_model.test(victim_test_loader, f"Epoch {epoch} Prune Test")
        if test_acc > best_acc:
            best_acc = test_acc
            pruner.export_model(model_path=f"{save_folder_pruned_victim}/best.pth",
                                mask_path=f"{save_folder_pruned_victim}/best_mask.pth")
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
        attack_test_loader = DataLoader(attack_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        
        shadow_model_path = f"{save_folder_original}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(shadow_model_path)
        
        org_state = copy.deepcopy(shadow_model.model.state_dict())
        pruned_shadow_model_save_folder = \
            f"{save_folder_pruned}/shadow_model_{shadow_ind}"
        if not os.path.exists(pruned_shadow_model_save_folder):
            os.makedirs(pruned_shadow_model_save_folder)
        
        shadow_pruned_model = BaseModel(args.dataset_name, args.model_name, t_max = args.t_max, 
                                 num_cls=args.num_cls, input_dim=args.input_dim, save_folder=pruned_shadow_model_save_folder, device=device, optimizer=args.optimizer, lr=prune_lr, weight_decay=args.weight_decay)
        
        shadow_pruned_model.model.load_state_dict(org_state)
        pruner = get_pruner(args.pruner_name, shadow_pruned_model.model, sparsity=args.prune_sparsity,)
        shadow_pruned_model.model = pruner.compress()

        best_acc = 0
        count = 0
        for epoch in range(args.prune_epochs):
            pruner.update_epoch(epoch)
            train_acc, train_loss = shadow_pruned_model.train(epoch, attack_train_loader, f"Epoch {epoch} Shadow Prune Train")
            test_acc, test_loss = shadow_pruned_model.test(attack_test_loader, f"Epoch {epoch} Shadow Prune Test")

            if test_acc > best_acc:
                best_acc = test_acc
                pruner.export_model(model_path=f"{pruned_shadow_model_save_folder}/best.pth",
                                    mask_path=f"{pruned_shadow_model_save_folder}/best_mask.pth")
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
