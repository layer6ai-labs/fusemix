import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class FeaturePairRetrievalDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_feat_path, text_feat_path):
        """
        vis_feat_path (string): path to tensor dict. file containing visual features
        text_feat_path (string): path to tensor dict. file containing text features
        """
        self.vis_feats = torch.load(vis_feat_path)
        self.text_feats = torch.load(text_feat_path)
        assert "txt2img" in self.text_feats.keys(), "missing mapping from text to visual features, ensure feature extraction was done correctly."
        assert "img2txt" in self.text_feats.keys(), "missing mapping from visual to text features, ensure feature extraction was done correctly."
        self.txt2img = self.text_feats.pop("txt2img")
        self.img2txt = self.text_feats.pop("img2txt")
        assert len(self.vis_feats) == len(self.img2txt), "length mismatch: there are {} visual features and {} visual mappings.".format(len(self.vis_feats), len(self.img2txt))
        assert len(self.text_feats) == len(self.txt2img), "length mismatch: there are {} text features and {} text mappings.".format(len(self.text_feats), len(self.txt2img))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.text_feats_list = []
        for i in range(len(self.text_feats)):
            text_embed = self.text_processor(self.text_feats[str(i)])
            self.text_feats_list.append(text_embed)

    def __len__(self):
        return len(self.vis_feats)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor 

    def __getitem__(self, index):
        vis_embed = self.vis_feats[str(index)]
        vis_embed = self.vis_processor(vis_embed)

        return {
            "vis_embed": vis_embed,
            "instance_id": str(index)
        }
