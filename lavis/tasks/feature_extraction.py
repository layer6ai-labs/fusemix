import os
import logging
from pathlib import Path
import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized 
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.logger import MetricLogger
from lavis.datasets.data_utils import prepare_sample
from lavis.common.utils import get_cache_path


@registry.register_task("feature_extraction")
class FeatureExtractionTask(BaseTask):
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

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.update(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def after_evaluation(self, val_result, split_name, **kwargs):
        self.save_result(
            result=val_result,
            result_dir=self.save_dir,
            filename=split_name,
        )

    @staticmethod
    def save_result(result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.dpt" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.dpt" % filename)

        torch.save(result, result_file)
        del result

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = {}
            
            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.dpt" % (filename, rank)
                )
                res = torch.load(result_file)
                for index in res:
                    result[index] = res[index]
                os.remove(result_file)
            
            torch.save(result, final_result_file)
            print("%d features saved to %s" % (len(result), final_result_file))

        return final_result_file
