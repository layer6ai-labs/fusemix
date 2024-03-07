import math

from lavis.common.registry import registry


@registry.register_mixup_alpha_scheduler("exp_mixup_alpha")
class ExpMixupAlphaScheduler:
    def __init__(
        self,
        model,
        max_epoch,
        init_alpha,
        max_alpha,
        **kwargs
    ):
        self.model = model
        assert hasattr(model, "mixup_alpha"), "This model does not support mixup."

        self.max_epoch = max_epoch
        self.init_alpha = init_alpha
        self.max_alpha = max_alpha
        assert init_alpha > 0 and max_alpha > 0

    def step(self, cur_epoch):
        exp_mixup_alpha_schedule(
            epoch=cur_epoch,
            model=self.model,
            max_epoch=self.max_epoch,
            init_alpha=self.init_alpha,
            max_alpha=self.max_alpha,
        )


@registry.register_mixup_alpha_scheduler("cosine_mixup_alpha")
class CosineMixupAlphaScheduler:
    def __init__(
        self,
        model,
        max_epoch,
        init_alpha,
        max_alpha,
        **kwargs
    ):
        self.model = model
        assert hasattr(model, "mixup_alpha"), "This model does not support mixup."
        
        self.max_epoch = max_epoch
        self.init_alpha = init_alpha
        self.max_alpha = max_alpha
        assert init_alpha > 0 and max_alpha > 0

    def step(self, cur_epoch):
        cosine_mixup_alpha_schedule(
            epoch=cur_epoch,
            model=self.model,
            max_epoch=self.max_epoch,
            init_alpha=self.init_alpha,
            max_alpha=self.max_alpha,
        )


def cosine_mixup_alpha_schedule(model, epoch, max_epoch, init_alpha, max_alpha):
    alpha = (init_alpha - max_alpha) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + max_alpha
    model.mixup_alpha = alpha

def exp_mixup_alpha_schedule(model, epoch, max_epoch, init_alpha, max_alpha):
    exp_rate = init_alpha / max_alpha
    alpha = init_alpha * exp_rate**(-epoch/max_epoch)
    model.mixup_alpha = alpha
