import torch
import math

class Sps(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.5,
                 gamma=2.0,
                 eta_max=None,
                 adapt_flag='smooth_iter',
                 fstar_flag=None,
                 eps=1e-8,
                 centralize_grad_norm=False,
                 centralize_grad=False):
        params = list(params)
        super().__init__(params, {})
        self.eps = eps
        self.params = params
        self.c = c
        self.centralize_grad_norm = centralize_grad_norm
        self.centralize_grad = centralize_grad

        if centralize_grad:
            assert self.centralize_grad_norm is False

        self.eta_max = eta_max
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.adapt_flag = adapt_flag
        self.state['step'] = 0

        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.fstar_flag = fstar_flag

    def step(self, closure=None, loss=None, batch=None):
        if loss is None and closure is None:
            raise ValueError('please specify either closure or loss')

        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)

        # increment step
        self.state['step'] += 1

        # get fstar
        if self.fstar_flag:
            fstar = float(batch['meta']['fstar'].mean())
        else:
            fstar = 0.

        # get loss and compute gradients
        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'if loss is provided then closure should beNone'

        # save the current parameters:
        grad_current = get_grad_list(self.params, centralize_grad=self.centralize_grad)
        grad_norm = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm)

        if grad_norm < 1e-8:
            step_size = 0.
        else:
            # adapt the step size
            if self.adapt_flag in ['constant']:
                # adjust the step size based on an upper bound and fstar
                step_size = (loss - fstar) / \
                    (self.c * (grad_norm)**2 + self.eps)
                if loss < fstar:
                    step_size = 0.
                else:
                    if self.eta_max is None:
                        step_size = step_size.item()
                    else:
                        step_size = min(self.eta_max, step_size.item())

            elif self.adapt_flag in ['smooth_iter']:
                # smoothly adjust the step size
                step_size = loss / (self.c * (grad_norm)**2 + self.eps)
                coeff = self.gamma**(1./self.n_batches_per_epoch)
                step_size = min(coeff * self.state['step_size'],
                                step_size.item())
            else:
                raise ValueError('adapt_flag: %s not supported' %
                                 self.adapt_flag)

            # update with step size
            sgd_update(self.params, step_size, grad_current)

        # update state with metrics
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        self.state['step_size'] = step_size
        self.state['grad_norm'] = grad_norm.item()

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('Got NaNs')

        return float(loss)

# utils
# ------------------------------
def compute_grad_norm(grad_list, centralize_grad_norm=False):
    grad_norm = 0.
    for g in grad_list:
        if g is None or (isinstance(g, float) and g == 0.):
            continue

        if g.dim() > 1 and centralize_grad_norm: 
            # centralize grads 
            g.add_(-g.mean(dim = tuple(range(1,g.dim())), keepdim = True))

        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params, centralize_grad=False):
    grad_list = []
    for p in params:
        g = p.grad
        if g is None:
            g = 0.
        else:
            g = p.grad.data
            if len(list(g.size()))>1 and centralize_grad:
                # centralize grads
                g.add_(-g.mean(dim = tuple(range(1,len(list(g.size())))), 
                       keepdim = True))
                   
        grad_list += [g]        
                   
    return grad_list


def sgd_update(params, step_size, grad_current):
    for p, g in zip(params, grad_current):
        if isinstance(g, float) and g == 0.:
            continue
        p.data.add_(other=g, alpha=- step_size)


class Chyps(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1e-3,
                 gamma=0.9,
                 epsilon=1e-8,
                 beta=10.0,
                 tau=2.0,
                 option='II', # 'I' or 'II'
                 centralize_grad=False):
        
        defaults = dict(init_step_size=init_step_size, gamma=gamma, 
                        epsilon=epsilon, beta=beta, tau=tau, option=option)
        super().__init__(params, defaults)
        
        self.params = self.param_groups[0]['params']
        self.centralize_grad = centralize_grad
        self.n_batches_per_epoch = n_batches_per_epoch
        
        # State initialization required for trainval.py compatibility
        self.state['step_size'] = init_step_size 
        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['grad_norm'] = 0.0
        self.state['gv_stats'] = {}
        
        # State initialization specific to CHYPS
        self.state['step'] = 0
        self.state['sigma_sq'] = 0.0
        self.state['lambda'] = init_step_size
        self.state['prev_grad_norm'] = 0.0
        self.state['x_prev'] = [p.data.clone().detach() for p in self.params]

    def step(self, closure=None, batch=None):
        if closure is None:
            raise ValueError('CHYPS requires a closure to evaluate gradients at multiple points.')

        self.state['step'] += 1
        k = self.state['step']
        group = self.param_groups[0]
        gamma = group['gamma']
        eps = group['epsilon']
        
        # 1. Evaluate standard gradient at x^k
        loss = closure()
        grad_k = self._get_grad_list()
        
        if k == 1:
            # Initialization step: x^1 = x^0 - \lambda_0 \nabla f_{\xi^0}(x^0)
            step_size = group['init_step_size']
            self.state['prev_grad_norm'] = self._compute_grad_norm(grad_k)
            self._update_params(step_size, grad_k)
            self.state['lambda'] = step_size
            
            # Metrics update for trainval.py compatibility at k=1
            self.state['step_size'] = step_size
            self.state['grad_norm'] = self.state['prev_grad_norm']
            self.state['n_forwards'] += 1
            self.state['n_backwards'] += 1
            
            return float(loss)

        # 2. Evaluate gradient at x^{k-1} on the SAME batch \xi^k
        # Temporarily store x^k and load x^{k-1}
        x_k_tensors = [p.data.clone() for p in self.params]
        for p, x_prev in zip(self.params, self.state['x_prev']):
            p.data.copy_(x_prev)
            
        self.zero_grad()
        closure() # Second forward/backward pass
        grad_k_minus_1 = self._get_grad_list()
        
        # Restore x^k
        for p, x_k in zip(self.params, x_k_tensors):
            p.data.copy_(x_k)

        # 3. Compute m_k = || \nabla f_{\xi^k}(x^k) - \nabla f_{\xi^k}(x^{k-1}) ||^2
        m_k = 0.0
        for g_k, g_k_minus_1 in zip(grad_k, grad_k_minus_1):
            if g_k is not None and g_k_minus_1 is not None:
                diff = g_k - g_k_minus_1
                m_k += torch.sum(torch.mul(diff, diff)).item()

        # 4. Compute t_k = || x^k - x^{k-1} || directly from tensors for absolute robustness
        t_k_sq = 0.0
        for x_k, x_prev in zip(x_k_tensors, self.state['x_prev']):
            diff = x_k - x_prev
            t_k_sq += torch.sum(torch.mul(diff, diff)).item()
        t_k = math.sqrt(t_k_sq)

        # 5. Moving averages
        self.state['sigma_sq'] = gamma * self.state['sigma_sq'] + (1 - gamma) * m_k
        sigma_sq_hat = self.state['sigma_sq'] / (1 - math.pow(gamma, k))

        # 6. Compute \lambda_k based on Option I or II
        proposed_lambda = t_k / (math.sqrt(sigma_sq_hat) + eps)
        
        if group['option'] == 'I':
            step_size = min(proposed_lambda, group['beta'])
        elif group['option'] == 'II':
            smoothing_factor = math.pow(group['tau'], 1.0 / self.n_batches_per_epoch)
            step_size = min(proposed_lambda, smoothing_factor * self.state['lambda'])
        else:
            raise ValueError("Invalid option. Choose 'I' or 'II'.")

        # 7. Update parameters: x^{k+1} = x^k - \lambda_k \nabla f_{\xi^k}(x^k)
        # Save current x^k (stored securely in x_k_tensors) as x^{k-1} for the NEXT iteration
        for x_prev, x_k in zip(self.state['x_prev'], x_k_tensors):
            x_prev.copy_(x_k)
            
        self.state['prev_grad_norm'] = self._compute_grad_norm(grad_k)
        self._update_params(step_size, grad_k)
        self.state['lambda'] = step_size
        
        # Metrics update for trainval.py compatibility
        self.state['step_size'] = step_size
        self.state['grad_norm'] = self.state['prev_grad_norm']
        self.state['n_forwards'] += 2
        self.state['n_backwards'] += 2

        return float(loss)

    def _get_grad_list(self):
        grad_list = []
        for p in self.params:
            g = p.grad
            if g is None:
                grad_list.append(torch.zeros_like(p.data))
            else:
                g_data = g.data.clone()
                if len(list(g_data.size())) > 1 and self.centralize_grad:
                    g_data.add_(-g_data.mean(dim=tuple(range(1, len(list(g_data.size())))), keepdim=True))
                grad_list.append(g_data)
        return grad_list

    def _compute_grad_norm(self, grad_list):
        grad_norm = 0.0
        for g in grad_list:
            grad_norm += torch.sum(torch.mul(g, g)).item()
        return math.sqrt(grad_norm)

    def _update_params(self, step_size, grad_list):
        for p, g in zip(self.params, grad_list):
            p.data.add_(g, alpha=-step_size)