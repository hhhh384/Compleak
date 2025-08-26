import argparse
import json
import numpy as np
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from dataset import get_dataset, get_imagenet
from attackers import MiaAttack


parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--compress_name', default="qat", type=str)
parser.add_argument('--compress_sparsity', default="int8", type=str)
parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=128, type=int)

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cpu_device = torch.device("cpu:0")
    cudnn.benchmark = True

    if args.compress_name == "qat":
         attack_device = cpu_device
    else:
        attack_device = device
    print(f"attack_device:{ attack_device}")

    save_folder_original = f"results/{args.dataset_name}_{args.model_name}"
    save_folder_compress = f"results_compress/{args.dataset_name}_{args.model_name}_{args.compress_name}_{args.compress_sparsity}"

    print(f"Save Folder: {save_folder_original}")


    trainset, testset = get_dataset(args.dataset_name)
    total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder_compress}/inference_data_index.pkl"
    
    with open(data_path, 'rb') as f:
        inference_victim_train_list, inference_victim_test_list, inference_attack_split_list = pickle.load(f)

    victim_train_dataset = Subset(total_dataset, inference_victim_train_list)
    victim_test_dataset = Subset(total_dataset, inference_victim_test_list)

    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    victim_model_save_folder = save_folder_original + "/victim_model"
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path)

    compress_victim_model_save_folder = save_folder_compress + "/victim_model"
    compress_victim_model_path = f"{compress_victim_model_save_folder}/best.pth"
    victim_compress_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=compress_victim_model_save_folder, device=attack_device)
    if args.compress_name == "qat":
        victim_compress_model.load_torchscript_model(compress_victim_model_path)
    else:
        victim_compress_model.model.load_state_dict(torch.load(f"{compress_victim_model_save_folder}/best.pth"))

    # Load shadow models
    shadow_model_list, shadow_compress_model_list, shadow_train_loader_list, shadow_test_loader_list = [], [], [], []
    for shadow_ind in range(args.shadow_num):
        attack_train_list, attack_test_list = inference_attack_split_list[shadow_ind]
        shadow_train_dataset = Subset(total_dataset, attack_train_list)
        shadow_test_dataset = Subset(total_dataset, attack_test_list)

        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        shadow_model_path = f"{save_folder_original}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(shadow_model_path)

        compress_shadow_model_save_folder = f"{save_folder_compress}/shadow_model_{shadow_ind}"
        compress_shadow_model_path = f"{compress_shadow_model_save_folder}/best.pth"
        shadow_compress_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=compress_shadow_model_save_folder, device= attack_device)
        if args.compress_name == "qat":
            shadow_compress_model.load_torchscript_model(compress_shadow_model_path)
        else:
            shadow_compress_model.model.load_state_dict(torch.load(f"{compress_shadow_model_save_folder}/best.pth"))

        shadow_model_list.append(shadow_model)
        shadow_compress_model_list.append(shadow_compress_model)
        shadow_train_loader_list.append(shadow_train_loader)
        shadow_test_loader_list.append(shadow_test_loader)


    print("Start Membership Inference Attacks")
    attacker = MiaAttack(
        victim_model, victim_compress_model, victim_train_loader, victim_test_loader,
        shadow_model_list, shadow_compress_model_list, shadow_train_loader_list, shadow_test_loader_list,
        num_cls=args.num_cls, device=device, batch_size=args.batch_size, save_folder = save_folder_compress)
    attacker.Compleak()

if __name__ == '__main__':
    args = parser.parse_args()

    print(args)
    main(args)
