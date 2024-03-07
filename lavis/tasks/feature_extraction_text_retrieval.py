import os
import logging
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils import data
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized 
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.logger import MetricLogger
from lavis.datasets.data_utils import prepare_sample
from lavis.common.utils import get_cache_path


@registry.register_task("feature_extraction_text_retrieval")
class FeatureExtractionTextRetrievalTask(BaseTask):
    def __init__(self, arch, model_type, dataset_name):
        super().__init__()

        save_dir = Path(get_cache_path(os.path.join("features", arch, model_type, dataset_name)))
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

    @classmethod
    def setup_task(cls, cfg):
        model_cfg = cfg.model_cfg
        datasets_cfg = cfg.datasets_cfg
        assert len(datasets_cfg) == 1, "Only one dataset can be provided at a time for feature extraction."
        
        arch = model_cfg.arch
        model_type = model_cfg.model_type
        dataset_name = list(datasets_cfg.keys())[0]

        return cls(
            arch=arch,
            model_type=model_type,
            dataset_name=dataset_name,
        )

    def train_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def train_iters(self, *args, **kwargs):
        raise NotImplementedError

    def valid_step(self, model, samples):
        results = {}
        feats = model.extract_features(samples)["embeds"].float()
        feats = feats.cpu().detach()
        indices = samples[self.inst_id_key]
        for feat, index in zip(feats, indices):
            if isinstance(index, torch.Tensor):
                index = index.item()
            results[index] = feat

        return results

    def evaluation(self, model, data_loader, cuda_enabled=True, **kwargs):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = {} # dictionary instead of list

        if is_main_process(): # since there's no distributed sampling here
            results["txt2img"] = data_loader.dataset.txt2img
            results["img2txt"] = data_loader.dataset.img2txt

            texts = data_loader.dataset.text
            num_text = len(texts)
            text_bs = 256

            for i in metric_logger.log_every(range(0, num_text, text_bs), print_freq, header):
                text = texts[i : min(num_text, i + text_bs)]
                samples = {"text_input": text, "instance_id": [str(idx) for idx in range(i, min(num_text, i + text_bs))]}

                eval_output = self.valid_step(model=model, samples=samples)
                results.update(eval_output)

        return results

    def after_evaluation(self, val_result, split_name, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=self.save_dir,
            filename=split_name,
        )

    @staticmethod
    def save_result(result, result_dir, filename):
        final_result_file = os.path.join(result_dir, "%s.dpt" % filename)

        if is_main_process():
            logging.warning("rank %d is now saving results." % get_rank()) 
            torch.save(result, final_result_file)
            # Subtract 2 since there are 2 non-feature keys
            print("%d text retrieval features saved to %s" % (len(result)-2 , final_result_file))

        return final_result_file
