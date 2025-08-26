import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, StepLR
import copy
from utils import get_model, get_train_transformer, WarmUpLR, get_optimizer, weight_init, get_optimizer
import torchvision.transforms as transforms
import torch.nn as nn

class BaseModel:
    def __init__(self, dataset_name, model_type, t_max=125, warm = 0, iter_per_epoch=0, device="cuda", save_folder="", num_cls=10,
                 optimizer="sgd", lr=0.1, weight_decay=5e-4, input_dim=100, epochs=0):
        
        self.model = get_model(model_type, num_cls, input_dim)
        self.model.to(device)
        self.device = device
        self.t_max = t_max
        self.dataset_name = dataset_name
        self.optimizer = get_optimizer(optimizer, self.model.parameters(), lr, weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.t_max)
        self.transform_train  = get_train_transformer(self.dataset_name)
        self.criterion = nn.CrossEntropyLoss()
        self.warm = warm
        self.iter_per_epoch = iter_per_epoch
        if self.warm:
            self.warmup_scheduler = WarmUpLR(self.optimizer, self.iter_per_epoch * self.warm)
        self.save_pref = save_folder
        self.num_cls = num_cls
        self.train_loss = []
        self.test_loss = []

    def cluster_train(self, epoch, trainloader):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        correct = 0
        total_loss = 0
        total = 0
        if self.warm:
            if epoch > self.warm:
                self.scheduler.step()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if self.dataset_name == "texas100":
                targets = torch.argmax((targets == 0).long(), dim=1)
            inputs, targets = inputs.cuda(), targets.cuda()
            if self.transform_train:
                inputs = torch.stack([self.transform_train(img) for img in inputs])
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            l2_reg = sum(torch.norm(param, 2) ** 2 for param in self.model.parameters()) / 2
            loss += 5e-5  * l2_reg
            loss.backward()
            for module in self.model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    for name, par in module.named_parameters():
                        if 'weight' in name:  
                            par.grad = torch.ones_like(par.grad) * par.grad
                            w = par.unique()
                            for i in range(len(w)):
                                mask = par == w[i]  
                                grads = par.grad[mask] 
                                new_shared_grad = grads.sum()  
                                par.grad[mask] = new_shared_grad.repeat(len(grads)) 
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            train_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            if self.warm:
                if epoch <= self.warm:
                    self.warmup_scheduler.step()
        if not self.warm:
            if self.scheduler:
                self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        print("{}: Accuracy {:.3f}, Loss {:.3f}".format(epoch, acc, total_loss))
        return acc, total_loss
    
    def train(self, epoch, train_loader, log_pref=""):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        if self.warm:
            if epoch > self.warm:
                self.scheduler.step()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device) 
            if self.transform_train:
                inputs = torch.stack([self.transform_train(img) for img in inputs])
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
          
            l2_reg = sum(torch.norm(param, 2) ** 2 for param in self.model.parameters()) / 2
            loss += 5e-5  * l2_reg
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            if self.warm:
                if epoch <= self.warm:
                    self.warmup_scheduler.step()
        if not self.warm:
            if self.scheduler:
                self.scheduler.step()
        acc = 100. * correct / total
        total_loss /= total
        self.train_loss.append(total_loss)  
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss

    def test(self, test_loader, log_pref=""):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                if isinstance(self.criterion, nn.BCELoss):
                    correct += torch.sum(torch.round(outputs) == targets)
                else:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        acc = 100. * correct / total
        total_loss /= total
        self.test_loss.append(total_loss)  
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc, total_loss))
        return acc, total_loss


    def save(self, epoch, acc, loss):
        save_path = f"{self.save_pref}/{epoch}.pth"
        state = {
            'epoch': epoch + 1,
            'acc': acc,
            'loss': loss,
            'state': self.model.state_dict()
        }
        torch.save(state, save_path)
        #torch.save(self.model._module.state_dict(), save_path)
        return save_path

    def load(self, load_path, verbose=False):
        state = torch.load(load_path, map_location=self.device)
        acc = state['acc']
        if verbose:
            print(f"Load model from {load_path}")
            print(f"Epoch {state['epoch']}, Acc: {state['acc']:.3f}, Loss: {state['loss']:.3f}")
        self.model.load_state_dict(state['state'])
        return acc


    def load_torchscript_model(self, load_path):
        cpu_device = torch.device("cpu:0")
        self.model = torch.jit.load(load_path, map_location=cpu_device)
        return cpu_device 

    def save_torchscript_qmodel(self, epoch):
        cpu_device = torch.device("cpu:0")
        q_model = copy.deepcopy(self.model)
        q_model.to(cpu_device)
        save_path = f"{self.save_pref}/{epoch}.pth"
        torch.jit.save(torch.jit.script(q_model), save_path)
        return save_path

    def save_torchscript_model(self, epoch):
        cpu_device = torch.device("cpu:0")
        q_model = copy.deepcopy(self.model)
        q_model.to(cpu_device)
        q_model = torch.quantization.convert(q_model, inplace=True)

        save_path = f"{self.save_pref}/{epoch}.pth"
        torch.jit.save(torch.jit.script(q_model), save_path)
        return save_path
    
    def predict_target(self, data_loader):
        self.model.eval()
        predict_list = []
        target_list = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predicts = F.softmax(outputs, dim=-1)
                predict_list.append(predicts.detach().data.cpu())
                target_list.append(targets)
     
        targets = torch.cat(target_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        return predicts, targets
    
    def predict_target_loss(self, data_loader, m=10, epsilon=1e-3):
        self.model  = self.model.to(self.device)
        self.model.eval()
        predict_list = []
        target_list = []
        loss_list = []
        success_list = []  
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.to(self.device)
                predicts = F.softmax(outputs, dim=-1)
                _, predicted_labels = torch.max(predicts, dim=1)
                success = (predicted_labels == targets).float()  

                predict_list.append(predicts.detach().data.cpu())
                target_list.append(targets)
                loss = criterion(outputs, targets)
                loss_list.extend(loss.cpu().numpy())
                success_list.append(success.detach().data.cpu())  
                    
        targets = torch.cat(target_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        loss = torch.tensor(loss_list).detach().cpu()
        success = torch.cat(success_list, dim=0)  
        
        return loss, targets, predicts, success