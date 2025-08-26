import torch
import torch.nn.functional as F
import pickle
from base_model import BaseModel
import pandas as pd
import numpy as np
from NR_SR import ConstructFeature, Attack
from torch.utils.data import TensorDataset, DataLoader

class MiaAttack:
    def __init__(self, victim_model, victim_compress_model, victim_train_loader, victim_test_loader,
                 shadow_model_list, shadow_compress_model_list, shadow_train_loader_list, shadow_test_loader_list,
                 num_cls=10, device="cuda", batch_size=128, save_folder = ''):
        self.victim_model = victim_model
        self.victim_compress_model = victim_compress_model
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.shadow_model_list = shadow_model_list
        self.shadow_compress_model_list = shadow_compress_model_list
        self.shadow_train_loader_list = shadow_train_loader_list
        self.shadow_test_loader_list = shadow_test_loader_list
        self.num_cls = num_cls
        self.device = device
        self.batch_size = batch_size
        self.save_folder = save_folder
        self._prepare()

    def _prepare(self):
        attack_original_in_predicts_list, attack_original_out_predicts_list = [], []
        attack_compress_in_predicts_list, attack_compress_out_predicts_list = [], []
        attack_in_targets_list = []
        attack_out_targets_list= []

        for shadow_model, shadow_compress_model, shadow_train_loader, shadow_test_loader in zip(
                self.shadow_model_list, self.shadow_compress_model_list, self.shadow_train_loader_list,
                self.shadow_test_loader_list):

            attack_original_in_predicts, attack_in_targets = shadow_model.predict_target(shadow_train_loader)
            attack_compress_in_predicts, _ = shadow_compress_model.predict_target(shadow_train_loader)

            attack_original_out_predicts, attack_out_targets = shadow_model.predict_target(shadow_test_loader)
            attack_compress_out_predicts, _ = shadow_compress_model.predict_target(shadow_test_loader)

            attack_original_in_predicts_list.append(attack_original_in_predicts)
            attack_compress_in_predicts_list.append(attack_compress_in_predicts)
            attack_in_targets_list.append(attack_in_targets)
            attack_out_targets_list.append(attack_out_targets)
            attack_original_out_predicts_list.append(attack_original_out_predicts)            
            attack_compress_out_predicts_list.append(attack_compress_out_predicts)

        self.attack_original_in_predicts = torch.cat(attack_original_in_predicts_list, dim=0)
        self.attack_original_out_predicts = torch.cat(attack_original_out_predicts_list, dim=0)
        self.attack_compress_in_predicts = torch.cat(attack_compress_in_predicts_list, dim=0)
        self.attack_compress_out_predicts = torch.cat(attack_compress_out_predicts_list, dim=0)
        
        self.attack_in_targets = torch.cat(attack_in_targets_list, dim=0)
        self.attack_out_targets = torch.cat(attack_out_targets_list, dim=0)


        self.victim_original_in_predicts, self.victim_in_targets = self.victim_model.predict_target(self.victim_train_loader)
        self.victim_original_out_predicts, self.victim_out_targets = self.victim_model.predict_target(self.victim_test_loader)
        self.victim_compress_in_predicts, _ = self.victim_compress_model.predict_target(self.victim_train_loader)
        self.victim_compress_out_predicts, _ = self.victim_compress_model.predict_target(self.victim_test_loader)


    def construct_feature(self, posterior_df):
        feature = ConstructFeature(posterior_df)
        for method in ['compleak_NR1', 'compleak_NR2', 'compleak_SR1', 'compleak_SR2']:
            feature.obtain_feature(method, posterior_df)

    def _save_posterior(self, posterior_df, save_path):
        pickle.dump(posterior_df, open(save_path, 'wb'))


    def _load_posterior(self, save_path):
        return pickle.load(open(save_path, 'rb'))

    def differential_attack(self):

        self.shadow_posterior_df = pd.DataFrame(columns=["original", "compress", "targets", "label"])
        self.attack_in_targets = F.one_hot(self.attack_in_targets, num_classes=self.num_cls).float()
        for index in range(self.attack_original_in_predicts.shape[0]):
            self.shadow_posterior_df.loc[len(self.shadow_posterior_df)] = [self.attack_original_in_predicts[index].cpu().reshape([1, -1]), self.attack_compress_in_predicts[index].cpu().reshape([1, -1]), self.attack_in_targets[index].cpu().reshape([1,-1]),1]
        self.attack_out_targets = F.one_hot(self.attack_out_targets, num_classes=self.num_cls).float()
        for index in range(self.attack_original_out_predicts.shape[0]):
            self.shadow_posterior_df.loc[len(self.shadow_posterior_df)] = [self.attack_original_out_predicts[index].cpu().reshape([1, -1]), self.attack_compress_out_predicts[index].cpu().reshape([1, -1]),self.attack_out_targets[index].cpu().reshape([1,-1]), 0]
        self.construct_feature(self.shadow_posterior_df)

        
        self.victim_posterior_df = pd.DataFrame(columns=["original", "compress","targets", "label"])
        self.victim_in_targets = F.one_hot(self.victim_in_targets, num_classes=self.num_cls).float()
        for index in range(self.victim_original_in_predicts.shape[0]):
            self.victim_posterior_df.loc[len(self.victim_posterior_df)] = [self.victim_original_in_predicts[index].cpu().reshape([1, -1]), self.victim_compress_in_predicts[index].cpu().reshape([1, -1]), self.victim_in_targets[index].cpu().reshape([1,-1]),1]
        self.victim_out_targets = F.one_hot(self.victim_out_targets, num_classes=self.num_cls).float()
        for index in range(self.victim_original_out_predicts.shape[0]):
            self.victim_posterior_df.loc[len(self.victim_posterior_df)] = [self.victim_original_out_predicts[index].cpu().reshape([1, -1]), self.victim_compress_out_predicts[index].cpu().reshape([1, -1]),self.victim_out_targets[index].cpu().reshape([1,-1]), 0]
    
        self.construct_feature(self.victim_posterior_df)
        results_df = pd.DataFrame()
        for attack_model_name in ['LR', 'DT', 'RF']:
            for method in ['compleak_NR1', 'compleak_NR2', 'compleak_SR1', 'compleak_SR2']:
                attack = Attack(attack_model_name, self.shadow_posterior_df, self.victim_posterior_df)
                train_acc, train_auc, train_prob,_ = attack.train_attack_model(method)
                test_acc, test_auc, pred,test_prob,test_low = attack.test_attack_model(method)
                
                results_df = pd.concat([results_df, pd.DataFrame({
                    "attack_model_name": [attack_model_name],
                    "method": [method],
                    "predictions": [list(pred)],  
                    "train_prob": [list(train_prob.flatten())],  
                    "test_prob": [list(test_prob.flatten())]  
                })])

        results_df.to_csv(f"prob_results.csv", index=False)
        
