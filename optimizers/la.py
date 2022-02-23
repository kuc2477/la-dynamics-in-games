import torch
from torch.optim import Optimizer
from collections import defaultdict


def _params(optimizer):
    params = []
    for g in optimizer.param_groups:
        for p in g['params']:
            params.append(p)
    return params


class Lookahead(Optimizer):
    '''
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    '''
    def __init__(self, x_optimizer, y_optimizer, k=5, alpha=0.5,
                 src='slow',
                 dst='fast',
                 start_after=0,
                 pullback_buffers=False,
                 pullback_momentum=None,
                 device='cuda'):
        '''
        :param optimizer:inner optimizer
        :param k (int): number of lookahead steps
        :param alpha(float): linear interpolation factor. 1.0 recovers the
            inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on
            interpolation update
        '''
        assert (src, dst) in (
            ('slow', 'fast'),
        ), 'Invalid interpolation type'
        assert (0.0 <= alpha <= 1.0), f'Invalid slow update rate: {alpha}'
        assert (1 <= k), f'Invalid lookahead steps: {k}'
        self.x_optimizer = x_optimizer
        self.y_optimizer = y_optimizer
        self.x_param_groups = self.x_optimizer.param_groups
        self.y_param_groups = self.y_optimizer.param_groups
        self.k = k
        self.k_counter = 0
        self.g_counter = 0
        self.alpha = alpha
        self.src = src
        self.dst = dst

        assert pullback_momentum in ('reset', 'pullback', None)
        self.pullback_momentum = pullback_momentum

        # states (slow) for each parameter
        self.state = defaultdict(dict)

        # optimizer parameters
        for g in self.x_optimizer.param_groups:
            for p in g['params']:
                param_state = self.state[p]

                # slow
                param_state['slow_params'] = \
                    torch.zeros_like(p.data).to(device)
                param_state['slow_params'].copy_(p.data)

        for g in self.y_optimizer.param_groups:
            for p in g['params']:
                param_state = self.state[p]

                # slow
                param_state['slow_params'] = \
                    torch.zeros_like(p.data).to(device)
                param_state['slow_params'].copy_(p.data)

    def __getstate__(self):
        return self.state_dict()

    def state_dict(self):
        return {
            'state': self.state,
            'k': self.k,
            'alpha': self.alpha,
            'src': self.src,
            'dst': self.dst,
            'k_counter': self.k_counter,
            'pullback_momentum': self.pullback_momentum,
        }

    def load_state_dict(self, state):
        self.state = state['state']
        self.k = state['k']
        self.alpha = state['alpha']
        self.src = state['src']
        self.dst = state['dst']
        self.k_counter = state['k_counter']
        self.pullback_momentum = state['pullback_momentum']

    def backup_and_load_slow(self):
        """
        Useful for performing evaluation on the slow weights (which typically
        generalize better)

        """
        for g in self.x_optimizer.param_groups:
            for p in g['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['slow_params'])
        for g in self.y_optimizer.param_groups:
            for p in g['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['slow_params'])

    def clear_and_load_backup(self):
        for g in self.x_optimizer.param_groups:
            for p in g['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']
        for g in self.y_optimizer.param_groups:
            for p in g['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def _interpolate(self, optimizer, src='slow', dst='fast'):
        assert dst == 'fast'
        for g in optimizer.param_groups:
            for p in g['params']:
                p_state = self.state[p]

                # interpolate between the parameters
                if src == 'slow' and dst == 'fast':
                    p.data\
                        .mul_(self.alpha)\
                        .add_((1. - self.alpha) * p_state['slow_params'].data)
                else:
                    raise ValueError('Invalid interpolation type')

                # update the slow parameter
                p_state['slow_params'].data.copy_(p.data)

                # interpolate the momentum
                if self.pullback_momentum == 'pullback' and self.src == 'slow':
                    m_fast = optimizer.state[p]['momentum_buffer']
                    m_slow = p_state.get('slow_momentum', 0)
                    optimizer.state[p]['momentum_buffer'] = \
                        m_fast.mul_(self.alpha).add_(
                            (1.0 - self.alpha) * m_slow
                        )
                    p_state['slow_momentum'] = \
                        optimizer.state[p]['momentum_buffer']

                # or reset the momentum
                elif self.pullback_momentum == 'reset':
                    optimizer.state[p]['momentum_buffer'] = \
                        torch.zeros_like(p.data)

    def step(self, closure=None):
        self.k_counter += 1
        self.g_counter += 1

        if self.k_counter < self.k:
            return

        self.k_counter = 0

        # update slow weights
        with torch.no_grad():
            self._interpolate(self.x_optimizer, src=self.src, dst=self.dst)
            self._interpolate(self.y_optimizer, src=self.src, dst=self.dst)
