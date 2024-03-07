import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class FeaturePairDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_feat_path, text_feat_path):
        """
        vis_feat_path (string): path to tensor dict. file containing visual features
        text_feat_path (string): path to tensor dict. file containing text features
        """
        self.vis_feats = torch.load(vis_feat_path)
        self.text_feats = torch.load(text_feat_path)
        assert len(self.vis_feats) == len(self.text_feats), "length mismatch: there are {} visual features and {} text features.".format(len(self.vis_feats), len(self.text_feats))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.vis_feats)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def sample_vis_k_dpp_subset(self, k):
        vis_feats_norm_np = np.empty((len(self), len(self.vis_feats["0"])))
        for i in range(len(self)):
            vis_feats_norm_np[i] = F.normalize(self.vis_feats[str(i)], dim=0).numpy()

        L = np.square(vis_feats_norm_np.dot(vis_feats_norm_np.T) + 1.)
        print("Done constructing vis kernel matrix")
        indices = dpp(L, k)
        print("Done sampling k-DPP")
        return indices

    def sample_text_k_dpp_subset(self, k):
        text_feats_norm_np = np.empty((len(self), len(self.text_feats["0"])))
        for i in range(len(self)):
            text_feats_norm_np[i] = F.normalize(self.text_feats[str(i)], dim=0).numpy()

        L = np.square(text_feats_norm_np.dot(text_feats_norm_np.T) + 1.)
        print("Done constructing text kernel matrix")
        indices = dpp(L, k)
        print("Done sampling k-DPP")
        return indices

    def sample_uniform_subset(self, k):
        rng = np.random.RandomState(1)
        indices = rng.choice(len(self), size=k, replace=False)
        return indices

    def __getitem__(self, index):
        vis_embed = self.vis_feats[str(index)]
        text_embed = self.text_feats[str(index)]

        vis_embed = self.vis_processor(vis_embed)
        text_embed = self.text_processor(text_embed)

        return {
            "vis_embed": vis_embed,
            "text_embed": text_embed,
            "instance_id": str(index)
        }


def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items
