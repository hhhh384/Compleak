import argparse
import copy
import numpy as np
import os
import pickle
import random
import torch
import time
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from base_model import BaseModel
from dataset import get_dataset, get_imagenet
from qat_dp import model_equivalence,QuantizedResNet18

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)


parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--early_stop', default=10, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="sgd", type=str)
parser.add_argument('--quantized_epochs', default=30, type=int)
parser.add_argument('--quantized_name', default='qat', type=str)
parser.add_argument('--sparsity', default="int8", type=str)

parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--t_max', default=30, type=int)



def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.benchmark = True

    quantized_lr = args.lr
    save_folder_original = f"results/{args.dataset_name}_{args.model_name}_1"
    save_folder_quantized = f"results_compress/{args.sub}_{args.dataset_name}_{args.model_name}_{args.quantized_name}_{args.sparsity}"

    print(f"Save Folder: {save_folder_quantized}")


    trainset, testset = get_dataset(args.dataset_name)
    
    total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder_original}/data_index.pkl"

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

    print("QAT Model")
    org_state = copy.deepcopy(victim_model.model.state_dict())
    save_folder_quantized_victim= f"{save_folder_quantized}/victim_model"
    if not os.path.exists(save_folder_quantized_victim):
        os.makedirs(save_folder_quantized_victim)
    quantized_victim_model = BaseModel(args.dataset_name, args.model_name, t_max = args.t_max, num_cls=args.num_cls, input_dim=args.input_dim, lr=quantized_lr,
        weight_decay=args.weight_decay, save_folder=save_folder_quantized_victim, device=device, optimizer=args.optimizer)
    
    quantized_victim_model.model.load_state_dict(org_state)
    quantized_victim_model.model.to(device)
    fused_victim_model = copy.deepcopy(quantized_victim_model)
    quantized_victim_model.model.train()
    fused_victim_model.model.train()
    if args.model_name == "resnet18":
        fused_victim_model.model = torch.ao.quantization.fuse_modules_qat(fused_victim_model.model,
                                                         [["conv1", "bn1", "relu"]],
                                                         inplace=True)
        for module_name, module in fused_victim_model.model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.ao.quantization.fuse_modules_qat(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]],inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.ao.quantization.fuse_modules_qat(sub_block, [["0", "1"]],inplace=True)

    elif args.model_name == "resnet50":
        fused_victim_model.model = torch.ao.quantization.fuse_modules_qat(fused_victim_model.model,[["conv1.0", "conv1.1", "conv1.2"]], inplace=True)
        for module_name, module in fused_victim_model.model.named_children():
            if module_name == "conv1":  
                continue
            if "conv" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    print(basic_block)
                    torch.ao.quantization.fuse_modules_qat( basic_block.residual_function, [["0", "1", "2"], ["3", "4","5"], [ "6", "7"]], inplace=True)
                    if hasattr(basic_block, "shortcut") and len(basic_block.shortcut) > 0:
                        torch.ao.quantization.fuse_modules_qat(basic_block.shortcut, [["0", "1"]], inplace=True)

    elif args.model_name == "vgg16":
        fused_victim_model.model.fuse_model(is_qat=True)
    elif args.model_name == "mobilenetv2":
        fused_victim_model.model.fuse_model(is_qat=True)


    # Print FP32 model.
    #print(quantized_victim_model.model)
    # Print fused model.
    print(fused_victim_model.model)
    # Model and fused model should be equivalent.
    quantized_victim_model.model.eval()
    fused_victim_model.model.eval()

    
    quantized_victim_model.model = QuantizedResNet18(model_fp32=fused_victim_model.model)
    victim_quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    #victim_quantization_config = torch.quantization.QConfig(activation=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver, dtype=torch.qint8),weight=torch.quantization.default_weight_fake_quant)
    quantized_victim_model.model.qconfig = victim_quantization_config

    # Print quantization configurations
    #print(quantized_victim_model.model.qconfig)
    # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
    torch.quantization.prepare_qat(quantized_victim_model.model, inplace=True)

    print("Training QAT Model...")
    best_acc = 0
    count = 0
    for epoch in range(args.quantized_epochs):
        train_acc, train_loss = quantized_victim_model.train(epoch, victim_train_loader, f"Epoch {epoch} QAT Train")
        test_acc, test_loss = quantized_victim_model.test(victim_test_loader, f"Epoch {epoch} QAT Test")
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = quantized_victim_model.save_torchscript_model(epoch)
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

        # load pretrained shadow model
        shadow_model_path = f"{save_folder_original}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.dataset_name, args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        
        shadow_model.load(shadow_model_path)

        org_state = copy.deepcopy(shadow_model.model.state_dict())
        quantized_shadow_model_save_folder = f"{save_folder_quantized}/shadow_model_{shadow_ind}"
        if not os.path.exists(quantized_shadow_model_save_folder):
            os.makedirs(quantized_shadow_model_save_folder)

    
        shadow_quantized_model = BaseModel(args.dataset_name, args.model_name, t_max = args.t_max, 
                                            num_cls=args.num_cls, input_dim=args.input_dim, lr=quantized_lr, weight_decay=args.weight_decay,
                                            save_folder=quantized_shadow_model_save_folder, device=device, optimizer=args.optimizer)
        
        shadow_quantized_model.model.load_state_dict(org_state)
        shadow_quantized_model.model.to(device)
    
        fused_shadow_model = copy.deepcopy(shadow_quantized_model)

        shadow_quantized_model.model.train()
        fused_shadow_model.model.train()
        if args.model_name == "resnet18_original":
            fused_shadow_model.model = torch.ao.quantization.fuse_modules_qat(fused_shadow_model.model,
                                                         [["conv1", "bn1", "relu"]],
                                                         inplace=True)
            for module_name, module in fused_shadow_model.model.named_children():
                if "layer" in module_name:
                    for basic_block_name, basic_block in module.named_children():
                        torch.ao.quantization.fuse_modules_qat(
                            basic_block, [["conv1", "bn1"], ["conv2", "bn2"]],
                            inplace=True)
                        for sub_block_name, sub_block in basic_block.named_children():
                            if sub_block_name == "downsample":
                                torch.ao.quantization.fuse_modules_qat(sub_block,
                                                               [["0", "1"]],
                                                               inplace=True)
                                
        elif args.model_name == "resnet50":
            fused_shadow_model.model = torch.ao.quantization.fuse_modules_qat(fused_shadow_model.model,
                                                         [["conv1.0", "conv1.1", "conv1.2"]], inplace=True)
        
            for module_name, module in fused_shadow_model.model.named_children():
                if module_name == "conv1": 
                    continue
                if "conv" in module_name:
                    for basic_block_name, basic_block in module.named_children():
                        print(basic_block)
                        torch.ao.quantization.fuse_modules_qat( basic_block.residual_function, [["0", "1", "2"], ["3", "4","5"], [ "6", "7"]], inplace=True)
                        if hasattr(basic_block, "shortcut") and len(basic_block.shortcut) > 0:
                            torch.ao.quantization.fuse_modules_qat(basic_block.shortcut, [["0", "1"]], inplace=True)
        elif args.model_name == "mobilenetv2":
            fused_shadow_model.model.fuse_model(is_qat=True)

        elif args.model_name == "vgg16":
            fused_shadow_model.model.fuse_model(is_qat=True)
    

        shadow_quantized_model.model.eval()
        fused_shadow_model.model.eval()

        shadow_quantized_model.model = QuantizedResNet18(model_fp32=fused_shadow_model.model)
        quantization_config = torch.quantization.get_default_qconfig("fbgemm")
        shadow_quantized_model.model.qconfig = quantization_config

        torch.quantization.prepare_qat(shadow_quantized_model.model, inplace=True)
        print("Training QAT shadow Model...")
        best_acc = 0
        count = 0
        for epoch in range(args.quantized_epochs):
            train_acc, train_loss = shadow_quantized_model.train(epoch, attack_train_loader, f"Epoch {epoch} QAT Train")
            test_acc, test_loss = shadow_quantized_model.test(attack_test_loader, f"Epoch {epoch} QAT Test")
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = shadow_quantized_model.save_torchscript_model(epoch)
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
