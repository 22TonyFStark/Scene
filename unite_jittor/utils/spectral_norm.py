"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import jittor
from jittor.misc import normalize
from typing import Any, Optional, TypeVar
from jittor.nn import Module


class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12): # return None
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: jittor.Var): # return jittor.Var
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool): # return jittor.Var
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with jittor.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(jittor.nn.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                    u = normalize(jittor.nn.matmul(weight_mat, v), dim=0, eps=self.eps)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = jittor.matmul(u, jittor.matmul(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module: Module): # return None
        with jittor.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        # module.register_parameter(self.name, jittor.Var(weight.detach()))
        setattr(module, self.name, jittor.Var(weight.detach()))

    def __call__(self, module: Module, inputs: Any): # return None
        self.compute_weight(module, do_power_iteration=module.is_training())
        # setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        # v = torch.linalg.multi_dot([weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]).squeeze(1)
        v = jittor.matmul(jittor.matmul(weight_mat.t().mm(weight_mat).pinverse(), jittor.matmul(weight_mat.t(), u.unsqueeze(1)))).squeeze(1)
        return v.mul_(target_sigma / jittor.matmul(u, jittor.matmul(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float): # return 'SpectralNorm'
        # for k, hook in module._forward_pre_hooks.items():
        #     if isinstance(hook, SpectralNorm) and hook.name == name:
        #         raise RuntimeError("Cannot register two spectral_norm hooks on "
        #                            "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        # if isinstance(weight, torch.nn.parameter.UninitializedParameter):
        #     raise ValueError(
        #         'The module passed to `SpectralNorm` can\'t have uninitialized parameters. '
        #         'Make sure to run the dummy forward before applying spectral normalization')

        with jittor.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            # u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            # v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
            u = normalize(jittor.randn([h]), dim=0, eps=fn.eps)
            v = normalize(jittor.randn([w]), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        # module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        # setattr(module, fn.name, weight.data)
        setattr(module, fn.name, weight)
        # module.register_buffer(fn.name + "_u", u)
        # module.register_buffer(fn.name + "_v", v)
        setattr(module, fn.name + "_u", u)
        setattr(module, fn.name + "_v", v)

        # module.register_forward_pre_hook(fn)
        module.register_pre_forward_hook(fn)
        # module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        # module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None): # return T_module
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(module, (jittor.nn.ConvTranspose,
                               jittor.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


# def remove_spectral_norm(module: T_module, name: str = 'weight') -> T_module:
#     r"""Removes the spectral normalization reparameterization from a module.

#     Args:
#         module (Module): containing module
#         name (str, optional): name of weight parameter

#     Example:
#         >>> m = spectral_norm(nn.Linear(40, 10))
#         >>> remove_spectral_norm(m)
#     """
#     for k, hook in module._forward_pre_hooks.items():
#         if isinstance(hook, SpectralNorm) and hook.name == name:
#             hook.remove(module)
#             del module._forward_pre_hooks[k]
#             break
#     else:
#         raise ValueError("spectral_norm of '{}' not found in {}".format(
#             name, module))

#     for k, hook in module._state_dict_hooks.items():
#         if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == name:
#             del module._state_dict_hooks[k]
#             break

#     for k, hook in module._load_state_dict_pre_hooks.items():
#         if isinstance(hook, SpectralNormLoadStateDictPreHook) and hook.fn.name == name:
#             del module._load_state_dict_pre_hooks[k]
#             break

#     return module
