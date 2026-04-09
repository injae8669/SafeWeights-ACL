import torch
import math

class MaskedAdamW(torch.optim.Optimizer):
    """
    Behavior in this implementation:
    - If masks is None or no mask is found for a parameter: behaves like standard AdamW.
    - If mask[name] exists:
        * Gradient moments (exp_avg/exp_avg_sq) and weight decay are computed as full AdamW.
        * The final parameter update is applied only where mask==1; mask==0 entries stay unchanged.
    """
    def __init__(self, params, masks=None, param_map=None,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

        self.masks = masks if masks is not None else {}
        # param_map: parameter tensor -> parameter name (str)
        self.param_map = param_map if param_map is not None else {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            beta1, beta2 = betas

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MaskedAdamW does not support sparse gradients")

                # 1) Fetch parameter mask if present.
                name = self.param_map.get(p, None)
                mask = None
                if name is not None and name in self.masks:
                    mask = self.masks[name].to(p.device)
                    if mask.dtype != p.dtype:
                        mask = mask.to(p.dtype)

                # 2) Initialize optimizer state exactly like AdamW.
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                step = state['step']

                beta1, beta2 = group['betas']

                # 3) Update first and second moments without masking.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4) Compute bias correction and denominator as in AdamW.
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                # 5) Build full AdamW update vector (including weight decay).
                # Decoupled AdamW form is equivalent to:
                #   p = p - lr * ( m_hat/denom + weight_decay * p )
                # We compute one update tensor, then apply masking only at the final step.
                update = exp_avg / denom
                if weight_decay != 0:
                    update = update + weight_decay * p.data

                # 6) Apply update with or without mask.
                if mask is None:
                    # Exactly AdamW behavior.
                    p.data.add_(update, alpha=-step_size)
                else:
                    # Update only where mask==1; keep mask==0 entries unchanged.
                    p.data.add_(update * mask, alpha=-step_size)

        return loss


