import copy
from scipy.spatial import distance
import numpy as np
import torch
import torch.nn.functional as F
from models import DT, RF, LR
import pandas as pd

class ConstructFeature:
    def __init__(self, posterior_df):
        self.posterior_df = posterior_df
    def obtain_feature(self, method, post_df):
        post_df = copy.deepcopy(post_df)
        self.posterior_df[method] = ""

        if method == "compleak_NR2_original":
            original_list = post_df.original.tolist()
            targets_list = post_df.targets.tolist()
            combined_data = [torch.cat((original, target), dim=1) for  original, target in zip(original_list, targets_list)]
            combined_series = pd.Series(combined_data)
            self.posterior_df[method] = combined_series

        elif method == "compleak_NR1_original":
            self.posterior_df[method] = self.posterior_df.original
        
        elif method == "compleak_NR1":
            self.posterior_df[method] = self.posterior_df.compress

        elif method == "compleak_NR2":
            compress_list = post_df.compress.tolist()
            targets_list = post_df.targets.tolist()
            combined_data = [torch.cat((compress, target), dim=1) for  compress, target in zip(compress_list, targets_list)]
            combined_series = pd.Series(combined_data)
            self.posterior_df[method] = combined_series

        elif method == "compleak_SR2":
            for index, posterior in enumerate(post_df.original):
                sort_indices = np.argsort(posterior[0, :])
                sort_indices = sort_indices.numpy()
                original_posterior = posterior[0, sort_indices].reshape((1, sort_indices.size))
                compress_posterior = post_df.compress[index][0, sort_indices].reshape((1, sort_indices.size))
                target = post_df.targets[index][0, sort_indices].reshape((1, sort_indices.size))
                conc = np.concatenate((original_posterior, compress_posterior), axis=1)
                conc = np.concatenate((conc, target), axis=1)
                self.posterior_df[method][index] = conc

        elif method == "compleak_SR1":
            for index, posterior in enumerate(post_df.original):
                sort_indices = np.argsort(posterior[0, :])
                sort_indices = sort_indices.numpy()
                original_posterior = posterior[0, sort_indices].reshape((1, sort_indices.size))
                compress_posterior = post_df.compress[index][0, sort_indices].reshape((1, sort_indices.size))
                conc = np.concatenate((original_posterior, compress_posterior), axis=1)
                self.posterior_df[method][index] = conc

        elif method == "l2_distance":
            for index in range(post_df.shape[0]):
                original_posterior = post_df.original[index][0]
                compress_posterior = post_df.compress[index][0]
                target = self.posterior_df.targets[index][0]
                target = target.reshape(1, -1)
                euclidean = distance.euclidean(original_posterior, compress_posterior)
                self.posterior_df[method][index] = np.full((1, 1), euclidean)
                self.posterior_df[method][index] = np.concatenate((self.posterior_df[method][index], target), axis=1)

        
        else:
            raise Exception("invalid feature construction method")


class Attack:
    def __init__(self, attack_model_name, shadow_post_df, target_post_df):

        self.attack_model_name = attack_model_name
        self.shadow_post_df = shadow_post_df
        self.target_post_df = target_post_df

        self.attack_model = self.determine_attack_model(attack_model_name)

    def determine_attack_model(self, attack_model_name):
        if attack_model_name == 'LR':
            return LR()
        elif attack_model_name == 'DT':
            return DT()
        elif attack_model_name == 'RF':
            return RF()
        else:
            raise Exception("invalid attack name")

    def train_attack_model(self, feature_construct_method):
        self.shadow_feature = self._concatenate_feature(self.shadow_post_df, feature_construct_method)
        label = self.shadow_post_df.label.astype('int')
        self.attack_model.train_model(self.shadow_feature, label)
        train_acc, _, prob = self.attack_model.test_model_acc(self.shadow_feature, label)
        train_auc, train_at_low = self.attack_model.test_model_auc(self.shadow_feature, label)
        print("attack model (%s, %s): train_acc:  %.3f | train_auc:  %.3f |train_at_low:  %.3f" % (self.attack_model_name, feature_construct_method, train_acc, train_auc, train_at_low))
        return train_acc, train_auc, prob, train_at_low

    def test_attack_model(self, feature_construct_method):
        self.target_feature = self._concatenate_feature(self.target_post_df, feature_construct_method)
        label = self.target_post_df.label.astype('int')
        test_acc, pred, prob = self.attack_model.test_model_acc(self.target_feature, label)
        test_auc, test_at_low = self.attack_model.test_model_auc(self.target_feature, label)
        print("attack model (%s, %s): test_acc:  %.3f | test_auc:  %.3f |test_at_low:  %.3f" % (self.attack_model_name, feature_construct_method, test_acc, test_auc, test_at_low))
        return test_acc, test_auc, pred, prob, test_at_low

    def obtain_attack_posterior(self, post_train, post_test, feature_construct_method):
        self.logger.info("obtaining attack posterior")
        post_train[feature_construct_method] = ""
        post_test[feature_construct_method] = ""
        post = self.attack_model.predict_proba(self.shadow_feature)
        for i in range(post.shape[0]):
            post_train.at[i, feature_construct_method] = post[i]
        post = self.attack_model.predict_proba(self.target_feature)
        for i in range(post.shape[0]):
            post_test.at[i, feature_construct_method] = post[i]


    def _concatenate_feature(self, posterior, method):
        feature = np.zeros((posterior[method][0].shape))

        for _, post in enumerate(posterior[method]):
            feature = np.concatenate((feature, post), axis=0)
        return feature[1:, :]