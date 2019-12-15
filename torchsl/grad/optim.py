from .functional import unitarize
from torch.optim.optimizer import Optimizer, required


class SPGD(Optimizer):

    def __init__(self, params, lr=required, restore_rate=5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if restore_rate < 1:
            raise ValueError("Invalid restore rate: {}".format(restore_rate))

        defaults = dict(lr=lr, restore_rate=restore_rate)
        super(SPGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_p.add_(-p.data @ d_p.t() @ p.data)
                n = group['lr'] * p.data.norm(1) / d_p.norm(1)
                p.data.add_(-n, d_p)

                param_state = self.state[p]
                if 'restore_counter' not in param_state:
                    param_state['restore_counter'] = 1
                else:
                    param_state['restore_counter'] += 1
                if param_state['restore_counter'] >= group['restore_rate']:
                    param_state['restore_counter'] -= group['restore_rate']
                    p.data = unitarize(p.data)
        return loss
