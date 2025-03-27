
import torch
from torch import nn
from copy import deepcopy

class EMA(nn.Module):
    """
    Exponential Moving Average:
    In deep refenforcement learning, the EMA is used as target the policy network.
    Update formula:
        ema_param = decay * ema_param + (1 - decay) * param
    """
    def __init__(self, model, decay=0.9999, update_after_step = 100, update_every = 10, inv_gamma = 1.0, power = 2 / 3):
        super(EMA, self).__init__()
        self.decay = decay
        self.model = model
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.inv_gamma = inv_gamma
        self.power = power
        self._init_emas()

        self.register_buffer('step', torch.tensor(0))

    
    def _init_emas(self):
        self.ema_model = deepcopy(self.model)

        for p in self.ema_model.parameters():
            p.detach_()

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}

    def add_to_optimizer_post_step_hook(self, optimizer):

        def hook(*args):
            self.update()

        return optimizer.register_step_post_hook(hook)
    
    def get_current_decay(self):
        actual_step = (self.step - self.update_after_step).clamp(min = 0.0)
        value = 1 - (1 + actual_step / self.inv_gamma) ** - self.power
        return value.clamp(min = 0.0, max = self.decay).item()


    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        training = self.ema_model.training
        self.ema_model.eval()
        out = self.ema_model(*args, **kwargs)
        self.ema_model.train(training)
        return out

    def eval(self):
        return self.ema_model.eval()

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def inplace_copy(tgt, src):
        tgt.copy_(src)

    def inplace_lerp(tgt, src, decay):
        tgt.lerp_(src, 1.0 - decay)

    def copy_params_from_model_to_ema(self):
        for (_, ema_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            self.inplace_copy(ema_params.data, current_params.data)

        for (_, ema_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            self.inplace_copy(ema_buffers.data, current_buffers.data)


    def copy_params_from_ema_to_model(self):
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            self.inplace_copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            self.inplace_copy(current_buffers.data, ma_buffers.data)

    def update(self):
        step = self.step.item()
        self.step += 1

        should_update = step % self.update_every == 0

        if should_update and step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if should_update:
            self.update_moving_average(self.ema_model, self.model)
        

    @torch.no_grad()
    def update_moving_average(self, ema_model, current_model, current_decay = None):
        if current_decay is None:
            current_decay = self.get_current_decay()

        tensors_to_lerp = []

        for (name, current_params), (_, ema_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ema_model)):
            tensors_to_lerp.append((ema_params.data, current_params.data))

        if len(tensors_to_lerp) > 0:
            tgt_lerp, src_lerp = zip(*tensors_to_lerp)
            torch._foreach_lerp_(tgt_lerp, src_lerp, 1.0 - current_decay)


        

        

        

