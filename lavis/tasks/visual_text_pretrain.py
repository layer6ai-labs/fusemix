import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.datasets.datasets.feature_pair_retrieval_dataset import FeaturePairRetrievalDataset
from lavis.models.fusion_models.utils import compute_image_retrieval_metrics


@registry.register_task("visual_text_pretrain")
class VisualTextPretrainTask(BaseTask):
    def __init__(self, tb_log_dir=None):
        super().__init__()
        self.writer = None
        if tb_log_dir is not None:
            self.writer = SummaryWriter(log_dir=tb_log_dir)

    @classmethod
    def setup_task(cls, cfg):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / cfg.run_cfg.output_dir / cfg.job_id
        tb_log_dir = output_dir / "tb_logs"
        tb_log_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("tb_log_dir", str(tb_log_dir))
        return cls(tb_log_dir=tb_log_dir)

    @torch.no_grad()
    def evaluation(self, model, data_loader, cur_epoch, **kwargs):

        if isinstance(data_loader.dataset, FeaturePairRetrievalDataset):
            score_i2t, score_t2i = model.compute_sim_matrix(data_loader)

            if is_main_process():
                eval_result = self._report_textimage_retrieval_metrics(
                    score_i2t,
                    score_t2i,
                    data_loader.dataset.txt2img,
                    data_loader.dataset.img2txt,
                )
                logging.info(eval_result)

                # summary tb
                if self.writer is not None and cur_epoch != "best":
                    for k, v in eval_result.items():
                        self.writer.add_scalar(f'test/{k}', v, cur_epoch)
                    self.writer.flush()
            else:
                eval_result = None

            return eval_result

        else:
            raise NotImplementedError

    # Used only for classification
    def valid_step(self, model, samples):
        results = []

        outputs = model.predict(samples)

        predictions = outputs["predictions"]
        targets = outputs["targets"]

        predictions = predictions.max(1)[1].cpu().numpy()
        targets = targets.cpu().numpy()

        indices = samples[self.inst_id_key]

        for pred, tgt, index in zip(predictions, targets, indices):
            if isinstance(index, torch.Tensor):
                index = index.item()

            results.append(
                {
                    self.inst_id_key: index,
                    "prediction": pred.item(),
                    "target": tgt.item(),
                }
            )

        return results

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    @staticmethod
    def _report_textimage_retrieval_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        return eval_result
