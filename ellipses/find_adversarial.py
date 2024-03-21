import os, torch
from tqdm import tqdm
from operators import l2_error
from typing import Callable, Optional, Tuple, List, Union, Type
import config
# ----- Variable Transforms -----
def identity(x):
    return x

def normalized_tanh(x, eps=1e-6):
    return (torch.tanh(x) + 1.0) / 2.0

def normalized_atanh(x, eps=1e-6):
    x = x * (1 - eps) * 2 - 1
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))

# ----- Optimization Methods -----
def PGD(
    loss        : Callable,
    t_in        : torch.Tensor,
    projs       : Union[List[Callable], Tuple[Callable]] = None,
    iter        : int   = 50,
    stepsize    : float = 1e-2,
    maxls       : int   = 50,
    ls_fac      : float = 0.1,
    ls_severity : float = 1.0,
    silent      : bool  = False,
):
    """ (Proj.) gradient descent with simple constraints.

    Minimizes a given loss function subject to optional constraints. The
    constraints must be "simple" in the sense that efficient projections onto
    the feasible set exist.

    The step size for gradient descent is determined by a backtracked
    line search. Set maximum number of line search steps to 1 to disable it.

    Parameters
    ----------
    loss : callable
        The loss or objective function.
    t_in : torch.Tensor
        The input tensor. This will be modified during
        optimization. The provided tensor serves as initial guess for the
        iterative optimization algorithm and will hold the result in the end.
    projs : list or tuple of callables, optional
        The projections onto the feasible set to perform after each gradient
        descent step. They will be performed in the order given. (Default None)
    iter : int, optional
        Number of iterations. (Default 50)
    stepsize : float, optional
        The step size parameter for gradient descent. Initial step size
        guess if line search is enabled. (Default 1e-2)
    maxls : int, optional
        Maximum number of line search steps. Set to 1 to disable the
        backtracked line search. (Default 10)
    ls_fac : float, optional
        Step size shrinkage factor for backtracked line search. Should be
        between 0 and 1. (Default 0.5)
    ls_severity : float, optional
        Line search severity parameter. Should be positive. (Default 1.0)
    silent : bool, optional
        Disable progress bar. (Default False)

    Returns
    -------
    torch.tensor
        The modified input t_in. Note that t_in is changed as a
        side-effect, even if the returned tensor is discarded.
    """

    def _project(t_in):
        with torch.no_grad():
            t_tmp = t_in.clone()
            if projs is not None:
                for proj in projs:
                    t_tmp = proj(t_tmp)
                t_in.data = t_tmp.data
            return t_tmp

    # run optimization
    ls_stepsize = stepsize
    t = tqdm(range(iter), desc="PGD iter", disable=silent)
    for it in t:
        """
        For each iteration the stepsize is initialised to stepsize of GD but
        an optimized stepsize is computed using a backtracking line search
        that decreases the stepsize until the Armijo-Goldstein condition is met.
        """
        # reset gradients
        if t_in.grad is not None:
            t_in.grad.detach_()
            t_in.grad.zero_()
        # compute pre step loss and descent direction
        pre_loss = loss(t_in)
        pre_loss.backward()
        p = -t_in.grad.data
        # backtracking line search : https://en.wikipedia.org/wiki/Backtracking_line_search
        ls_count, STOP_LS = 1, False
        with torch.no_grad():
            while not STOP_LS:
                # P(delta_i-1 + lambda * nabla_delta l(Psi_theta(y + delta_i-1), x)
                # P - projection operator found in _project
                # delta_i-1 - adversarial noise at iteration i-1
                # lambda - learning rate, in backtracking LS it's the stepsize ls_stepsize
                #  note that ls_stepsize is set initially to the GD stepsize
                # nabla_delta l(Psi_theta(y + delta_i-1), x) - gradient of loss wrt. adversarial noise of 
                #  reconstruction of perturbed image Psi_theta(y + delta_i-1) and ground truth image x
                t_tmp = _project(t_in + ls_stepsize * p)
                step_loss = loss(t_tmp)
                STOP_LS = (ls_count >= maxls) or (
                    (
                        pre_loss
                        - ls_severity * (p * (t_tmp - t_in)).mean()
                        + 1 / (2 * ls_stepsize) * (t_tmp - t_in).pow(2).mean()
                    )
                    > step_loss
                )
                if not STOP_LS:
                    ls_count += 1
                    ls_stepsize *= ls_fac
        # do the actual projected gradient step
        t_in.data.add_(ls_stepsize, p)
        _project(t_in)
        t.set_postfix(
            loss=step_loss.item(), ls_steps=ls_count, stepsize=ls_stepsize,
        )
        # allow initial step size guess to grow between iterations
        if ls_count < maxls and maxls > 1:
            ls_stepsize /= ls_fac
        # stop if steps become too small
        if ls_stepsize < 1e-18:
            break

    return t_in


def PAdam(
    loss     : Callable, 
    t_in     : torch.Tensor, 
    projs    : Union[List[Callable], Tuple[Callable]] = None, 
    niter    : int = 50, 
    stepsize : float = 1e-2, 
    silent   : bool =False,
) -> torch.Tensor:
    """ (Proj.) Adam accelerated gradient decent with simple constraints.

    Minimizes a given loss function subject to optional constraints. The
    constraints must be "simple" in the sense that efficient projections onto
    the feasible set exist.

    Parameters
    ----------
    loss : callable
        The loss or objective function.
    t_in : torch.Tensor
        The input tensor. This will be modified during
        optimization. The provided tensor serves as initial guess for the
        iterative optimization algorithm and will hold the result in the end.
    projs : list or tuple of callables, optional
        The projections onto the feasible set to perform after each gradient
        descent step. They will be performed in the order given. (Default None)
    iter : int, optional
        Number of iterations. (Default 50)
    stepsize : float, optional
        The step size parameter for gradient descent. (Default 1e-2)
    silent : bool, optional
        Disable progress bar. (Default False)

    Returns
    -------
    torch.tensor
        The modified input t_in. Note that t_in is changed as a
        side-effect, even if the returned tensor is discarded.
    """

    def _project(t_in):
        with torch.no_grad():
            t_tmp = t_in.clone()
            if projs is not None:
                # apply chain of projections
                for proj in projs:
                    t_tmp = proj(t_tmp)
                t_in.data = t_tmp.data
            return t_tmp

    # run optimization
    optimizer = torch.optim.Adam((t_in,), lr=stepsize, eps=1e-5)
    t = tqdm(range(niter), desc="PAdam iter", disable=silent)
    for it in t:
        # reset gradients
        if t_in.grad is not None:
            t_in.grad.detach_()
            t_in.grad.zero_()
        #  compute loss and take gradient step
        pre_loss = loss(t_in)
        pre_loss.backward()
        optimizer.step()
        # project and evaluate
        _project(t_in)
        post_loss = loss(t_in)
        t.set_postfix(pre_loss=pre_loss.item(), post_loss=post_loss.item())

    return t_in

from functools import partial
def PAdam_DIP_x(
    loss         : Callable,
    t_in         : torch.Tensor, 
    xhat0        : torch.Tensor,
    projs        : Union[List[Callable], Tuple[Callable]] = None, 
    niter        : int      = 50, 
    stepsize     : float    = 1e-2, 
    silent       : bool     = False,
) -> torch.Tensor:
    """ (Proj.) Adam accelerated gradient decent with simple constraints.

    Minimizes a given loss function subject to optional constraints. The
    constraints must be "simple" in the sense that efficient projections onto
    the feasible set exist.

    Parameters
    ----------
    loss : callable
        The loss or objective function.The revonstructed image xhat 
        will be updated during the optimization process so this is a dynamic argument.
        Loss function should be on the form:
          l(delta = t_in, x_0 = Psi_theta(z_tilde), x = x) = ||Axhat - A(x+delta)|| - beta||xhat - x||
    t_in : torch.Tensor
        The input tensor representing the adv. noise. This will be modified during
        optimization. The provided tensor serves as initial guess for the
        iterative optimization algorithm and will hold the result in the end.
    xhat0  : torch.Tensor
        Initial condition for first step in two-step optimization process for finding adversarial noise for DIP.
        Usually x_0 = Psi_theta(z_tilde) (output of the pre-trained DIP model)
    projs : list or tuple of callables, optional
        The projections onto the feasible set to perform after each gradient
        descent step. They will be performed in the order given. (Default None)
    niter : int, optional
        Number of iterations. (Default 50)
    stepsize : float, optional
        The step size parameter for gradient descent. (Default 1e-2)
    silent : bool, optional
        Disable progress bar. (Default False)

    Returns
    -------
    torch.tensor
        The modified input t_in. Note that t_in is changed as a
        side-effect, even if the returned tensor is discarded.
    """

    def _project(t_in):
        with torch.no_grad():
            t_tmp = t_in.clone()
            if projs is not None:
                # apply chain of projections
               for proj in projs:
                    t_tmp = proj(t_tmp)
               t_in.data = t_tmp.data
        return t_tmp
    # clone and clear prev. computational graph of xhat
    xhat = xhat0.clone().detach()
    xhat.requires_grad = True
    # init optimizer for reconstructed image xhat
    optimizer_xhat = torch.optim.Adam((xhat,), lr=stepsize, eps=1e-5)
    # init optimizer for adversarial noise
    optimizer = torch.optim.Adam((t_in,), lr=stepsize, eps=1e-5)
    t = tqdm(range(niter), desc="PAdam iter", disable=silent)
    for it in t:
        # ------------- xhat training step ------------------------------
        # enable autograd on xhat and disable on t_in
        xhat.requires_grad  = True
        t_in.requires_grad = False
        optimizer_xhat.zero_grad()
        # update the net parameters minimizing the DIP loss function
        # TODO: implement _closure  s.t. loss function can handle both xhat and t_in as arguments
        xhat_loss = loss(t_in, xhat)
        xhat_loss.backward()
        # update dip net parameter
        optimizer_xhat.step()
        
        # ------------- Adv. noise PGD step ------------------------------
        t_in.requires_grad = True
        xhat.requires_grad = False
        # reset gradients of adv. noise tensor
        if t_in.grad is not None:
            t_in.grad.detach_()
            t_in.grad.zero_()
        # make partial loss with frozen xhat s.t. only t_in is updated
        loss_adv_noise = partial(loss, xhat = xhat)
        #  compute loss and take gradient step
        # pre loss is loss before projection onto lp-ball
        pre_loss = loss_adv_noise(t_in)
        pre_loss.backward()
        optimizer.step()
        # project and evaluate
        _project(t_in)
        # post loss is loss after projection onto lp-ball
        post_loss = loss_adv_noise(t_in)
        t.set_postfix(
            pre_loss=pre_loss.item(), 
            post_loss=post_loss.item(), 
            l2norm_tin=t_in.pow(2).sum().item()
        )
    return t_in

def PAdam_DIP_theta(
    loss         : Callable,
    net          : Type[torch.nn.Module],
    dip_optimizer: Type[torch.optim.Optimizer],
    t_in         : torch.Tensor,
    projs        : Union[List[Callable], Tuple[Callable]] = None,
    iter         : int      = 50,
    stepsize     : float    = 1e-2,
    silent       : bool     = False,
) -> torch.Tensor:
    """ (Proj.) Adam accelerated gradient decent with simple constraints.

    Minimizes a given loss function subject to optional constraints. The
    constraints must be "simple" in the sense that efficient projections onto
    the feasible set exist.

    Parameters
    ----------
    loss : callable
        The loss or objective function. z_tilde should be given outside since this is a static argument.
        The net parameters theta will be updated during the optimization process so this is a dynamic argument.
    net  : torch.nn.Module 
        The reconstruction network.
    train_params: dict
        Training parameters containing e.g. DIP net optimizer.
    y0   : torch.Tensor
        Noiseless measurement.
    z_tilde : torch.Tensor
        The fixed random input tensor. Form Ulyanov et al. 2017 z_tilde ~ U([0,1/10]).
    t_in : torch.Tensor
        The input tensor representing the adv. noise. This will be modified during
        optimization. The provided tensor serves as initial guess for the
        iterative optimization algorithm and will hold the result in the end.
    projs : list or tuple of callables, optional
        The projections onto the feasible set to perform after each gradient
        descent step. They will be performed in the order given. (Default None)
    iter : int, optional
        Number of iterations. (Default 50)
    stepsize : float, optional
        The step size parameter for gradient descent. (Default 1e-2)
    silent : bool, optional
        Disable progress bar. (Default False)

    Returns
    -------
    torch.tensor
        The modified input t_in. Note that t_in is changed as a
        side-effect, even if the returned tensor is discarded.
    """
    def _project(t_in):
        with torch.no_grad():
            t_tmp = t_in.clone()
            if projs is not None:
                # apply chain of projections
                for proj in projs:
                    t_tmp = proj(t_tmp)
                t_in.data = t_tmp.data
            return t_tmp

    # run optimization
    optimizer = torch.optim.Adam((t_in,), lr=stepsize, eps=1e-5)
    t = tqdm(range(iter), desc="PAdam iter", disable=silent)
    for it in t:
        # ------------- DIP training step ------------------------------
        net.train()
        # reset gradients
        dip_optimizer.zero_grad()
        # update the net parameters minimizing the DIP loss function
        dip_loss = loss(t_in, net)
        dip_loss.backward()
        # update dip net parameter
        dip_optimizer.step()

        # ------------- Adv. noise PGD step ------------------------------
        # make sure only adv. noise is updatet in this step
        # turn of training mode in specific layers, fex. dropout
        net.eval()
        # fix parameters in dip net
        for param in net.parameters():
            param.requires_grad = False
        # reset gradients of adv. noise tensor
        if t_in.grad is not None:
            t_in.grad.detach_()
            t_in.grad.zero_()
        # make partial loss with the frozen net
        loss_adv_noise = partial(loss, net = net)
        #  compute loss and take gradient step
        # pre loss is loss before projection onto lp-ball
        pre_loss = loss_adv_noise(t_in)
        pre_loss.backward()
        optimizer.step()
        # project and evaluate
        _project(t_in)
        # post loss is loss after projection onto lp-ball
        post_loss = loss_adv_noise(t_in)
        t.set_postfix(pre_loss=pre_loss.item(), post_loss=post_loss.item())
    return t_in

# ----- Adversarial Example Finding -----
def untargeted_attack(
    func              : Callable,
    t_in_adv          : torch.Tensor,
    t_in_ref          : torch.Tensor,
    t_out_ref         : torch.Tensor = None,
    domain_dist       : Callable     = None,
    mixed_dist        : Callable     = None,
    codomain_dist     : Callable     = torch.nn.MSELoss(),
    weights           : Tuple        = (1.0, 1.0, 1.0),
    optimizer         : Callable     = PGD,
    transform         : Callable     = identity,
    inverse_transform : Callable     = identity,
    **kwargs
):
    """ Untargeted finding of adversarial examples.

    Finds perturbed input to a function f(x) that is close to a specified
    reference input and brings the function value f(x) as far from f(reference)
    as possible, by solving

        min distance(x, reference) - distance(f(x), f(reference))

    subject to optional constraints (see e.g. `PGD`). Optionally the
    optimization domain can be transformed to include implicit constraints.
    In this case a variable transform and its inverse must be provided.


    Parameters
    ----------
    func : callable
        The function f. For image reconstruction it's the function that
        reconstructs the image.
        Supervised : Psi_theta(y)
        DIP        : Psi_theta(z_tilde) for theta = argmin_theta ||A Psi_theta(z_tilde) - y ||_2^2 
    t_in_adv : torch.Tensor
        The adversarial input tensor. This will be modified during
        optimization. The provided tensor serves as initial guess for the
        iterative optimization algorithm and will hold the result in the end.
    t_in_ref : torch.tensor
        The reference input tensor.
    domain_dist : callable, optional
        The distance measure between x and reference in the domain of f. Set
        to `Ǹone` to exlcude this term from the objective. (Default None)
    codomain_dist : callable, optional
        The distance measure between f(x) and f(reference) in the codomain of
        f. (Default torch.nn.MSELoss)
    weights : tuple of float
        Weighting factor for the distance measures in the objective.
        (Default (1.0, 1.0, 1.0))
    optimizer : callable, optional
        The routine used for solving the optimization problem. (Default `PGD`i)
        Typically have the structure optimizer(loss, t_in, **kwargs)
    transform : callable, optional
        Domain variable transform. (Default `identity`)
    inverse_transform : callable, optional
        Inverse domain variable transform. (Default `identity`)
    **kwargs
        Optional keyword arguments passed on to the optimizer (see e.g. `PGD`).

    Returns
    -------
    torch.tensor
        The perturbed input t_in_adv. Note that t_in_adv is changed as a
        side-effect, even if the returned tensor is discarded.
    """

    t_in_adv.data = inverse_transform(t_in_adv.data)
    if t_out_ref is None:
        # if GT out is not given compute rec. from noiseless meas.
        t_out_ref = func(t_in_ref)

    # loss closure
    def _closure(t_in, xhat = None, func = func, codomain_dist = codomain_dist):
        if isinstance(func, partial):
            if "xhat" in func.func.__code__.co_varnames:
                func = partial(func, xhat = xhat) 
        # 1. Apply transform to perturbed measurment 
        # 2. Reconstruct image using func 
        t_out = func(transform(t_in))
        # insert necessary variables for DIP
        
        if isinstance(codomain_dist, partial):
            if "adv_example" in codomain_dist.func.__code__.co_varnames:
                # make a lambda function s.t. the only two arguments in the codomain_dist function 
                # is xhat and x IN that order
                new_cdd = codomain_dist
                def codomain_dist(xhat,x,t_in=t_in): 
                    return new_cdd(t_in, xhat,x)
        # computing codomain distance between reconstructed t and reference output
        # i.e. rec. image from perturbed meas. xhat and GT image x0
        # t_out     - reconstructed image
        # t_out_rec - GT image
        loss = -weights[1] * codomain_dist(t_out, t_out_ref)
        if domain_dist is not None:
            loss += weights[0] * domain_dist(transform(t_in), t_in_ref)
        if mixed_dist is not None:
            loss += weights[2] * mixed_dist(
                transform(t_in), t_in_ref, t_out, t_out_ref
            )
        return loss

    # optimizer is a function on the form optimizer(loss_function, t_in, **kwargs)
    t_in_adv = optimizer(_closure, t_in_adv, **kwargs)
    return transform(t_in_adv)


# ----- Grid attacks -----
def err_measure_l2(x1, x2):
    """ L2 error wrapper function. """
    return l2_error(x1, x2, relative=True, squared=False)[1].squeeze()


def grid_attack(
    method      : dict,
    noise_rel   : torch.Tensor,
    X_0         : torch.Tensor, 
    Y_0         : torch.Tensor,
    #rec_config  : dict     = None,
    store_data  : bool     = False,
    keep_init   : int      = 0,
    err_measure : Callable = err_measure_l2,
):
    """ Finding adversarial examples over a grid of multiple noise levels.

    Find adversarial examples for a reconstruction setup.

    Parameters
    ----------
    method : dataframe
        The reconstruction method (including metadata in a dataframe).
    noise_rel : torch.Tensor
        List of relative noise levels. An adversarial example will be computed
        for each noise level, in descending order.
    X_0 : torch.Tensor
        The reference signal.
    Y_0 : torch.Tensor
        The reference measurements. Used to convert relative noise levels to
        absolute noise levels.
    store_data : bool, optional
        Store resulting adversarial examples. If set to False, only the
        resulting errors are stored. (Default False)
    keep_init : int, optional
        Reuse results from one noise level as initialization for the next noise
        level. (Default 0)
    err_measure : callable, optional
        Error measure to evaluate the effect of the adversarial perturbations
        on the reconstruction method. (Default relative l2-error)

    Returns
    -------
    torch.Tensor
        Error of adversarial perturbations for each noise level.
    torch.Tensor
        Error of statistical perturbations for each noise level (as reference).
    torch.Tensor, optional
        The adversarial reconstruction for each noise level.
        (only if store_data is set to True)
    torch.Tensor, optional
        The reference reconstruction for each noise level.
        (only if store_data is set to True)
    torch.Tensor, optional
        The adversarial measurements for each noise level.
        (only if store_data is set to True)
    torch.Tensor, optional
        The reference measurements for each noise level.
        (only if store_data is set to True)

    """

    X_adv_err = torch.zeros(len(noise_rel), X_0.shape[0])
    X_ref_err = torch.zeros(len(noise_rel), X_0.shape[0])
    print('att: X_adv_err.shape: ', X_adv_err.shape)
    if store_data:
        X_adv = torch.zeros(
            len(noise_rel), *X_0.shape, device=torch.device("cpu")
        )
        X_ref = torch.zeros(
            len(noise_rel), *X_0.shape, device=torch.device("cpu")
        )

        Y_adv = torch.zeros(
            len(noise_rel), *Y_0.shape, device=torch.device("cpu")
        )
        Y_ref = torch.zeros(
            len(noise_rel), *Y_0.shape, device=torch.device("cpu")
        )
    # for DIP networks we need to learn pre-trained network
    if "DIP" in method.name:
        z_tilde = 0.1 * torch.rand( (1, method.rec_config["net_in_channels"],) + X_0.shape[-2:] )
        # save initialised model weights to be loaded in each reconstruction at each noise level
        torch.save(method["net"].state_dict(), os.path.join(config.RESULTS_PATH, "DIP_x_UNet_init_adv_table.pt") )
        # note that Y0 is Y0.shape[0] identical samples
        Xhat    = method.reconstr(y0 = Y_0[:1], net = method["net"], z_tilde = z_tilde)
        method.rec_config["xhat0"] = Xhat.repeat( Y_0.shape[0], *((X_0.ndim -1) * (1,)) )
     
    # for DeepDecoder networks we need to learn pre-trained network
    if "DeepDecoder" in method.name:
        # use same input name convention as for DIP to save space - TODO: use this same convention when defining DeepDecoder in the thesis
        z_tilde = 10 * torch.rand( (1, method.rec_config["net_in_channels"],) + tuple([d//2**(method["net"].nscales-1) for d in X_0.shape[-2:]]) )
        # save initialised model weights to be loaded in each reconstruction at each noise level
        torch.save(method["net"].state_dict(), os.path.join(config.RESULTS_PATH, "DeepDecoder_init_adv_table.pt") )
        # note that Y0 is Y0.shape[0] identical samples
        Xhat    = method.reconstr(y0 = Y_0[:1], net = method["net"], z_tilde = z_tilde)
        method.rec_config["xhat0"] = Xhat.repeat( Y_0.shape[0], *((X_0.ndim -1) * (1,)) )


    for idx_noise in reversed(range(len(noise_rel))):
        # perform the actual attack for "method" and current noise level
        print(
            "Method: "
            + method.name
            + "; Noise rel {}/{}".format(idx_noise + 1, len(noise_rel)),
            flush=True,
        )

        # compute adversarial examples
        if (keep_init == 0) or (idx_noise == (len(noise_rel) - 1)):
            Y_adv_cur, Y_ref_cur, Y_0_cur = method.attacker(
                X_0, noise_rel[idx_noise], yadv_init = None, rec_config = method.rec_config,
            )
        else:
            Y_adv_cur, Y_ref_cur, Y_0_cur = method.attacker(
                X_0, noise_rel[idx_noise], yadv_init = Y_adv_cur, rec_config = method.rec_config,
            )

        # if DIP method net and z_tilde have to be added as an argument to the reconstruction function
        if "DIP" in method.name:
            assrt_msg = "z_tilde not an argument in the DIP reconstruction function (note diff. form adv. rec. func)"
            assert "z_tilde" in method.reconstr.func.__code__.co_varnames, assrt_msg
            # re-initialize model at zero noise level with the same weights each adv. attack
            params_init = torch.load(os.path.join(config.RESULTS_PATH, "DIP_x_UNet_init_adv_table.pt"))
            method["net"].load_state_dict(params_init)
            # update reconstruction function with re-initialised model and the same z_tilde as for every reconstruction
            method["reconstr"] = partial(method.reconstr, net = method["net"], z_tilde=z_tilde)

        
        # if DeepDecoder method net and z_tilde have to be added as an argument to the reconstruction function
        if "DeepDecoder" in method.name:
            assrt_msg = "z_tilde not an argument in the DeepDecoder reconstruction function (note diff. form adv. rec. func)"
            assert "z_tilde" in method.reconstr.func.__code__.co_varnames, assrt_msg
            # re-initialize model at zero noise level with the same weights each adv. attack
            params_init = torch.load(os.path.join(config.RESULTS_PATH, "DeepDecoder_init_adv_table.pt"))
            method["net"].load_state_dict(params_init)
            # update reconstruction function with re-initialised model and the same z_tilde as for every reconstruction
            method["reconstr"] = partial(method.reconstr, net = method["net"], z_tilde=z_tilde)
        
        # compute adversarial and reference reconstruction
        # (noise level needs to be absolute)
        # DIP_x : this reconstructs the image xhat from measurements Y_adv_cur and Y_ref_cur
        X_adv_cur = method.reconstr(Y_adv_cur)#, noise_rel[idx_noise])
        # re-init net before reconstruction of reference measurement
        if "DIP" in method.name or "DeepDecoder" in method.name:
            method["net"].load_state_dict(params_init)
        
        X_ref_cur = method.reconstr(Y_ref_cur)#, noise_rel[idx_noise])
        
        # compute resulting reconstruction error according to err_measure
        shape_x0_repeat = (1,)*(X_0.ndim)
        if "DIP" in method.name:
            shape_x0_repeat = X_0.shape[:1] + (1,)*(X_0.ndim-1)
        X_adv_err[idx_noise, ...] = err_measure(X_adv_cur.repeat(*shape_x0_repeat), X_0)
        X_ref_err[idx_noise, ...] = err_measure(X_ref_cur.repeat(*shape_x0_repeat), X_0)
        print('X_adv_cur.shape: ', X_adv_cur.shape)

        if store_data:
            X_adv[idx_noise, ...] = X_adv_cur.cpu()
            X_ref[idx_noise, ...] = X_ref_cur.cpu()

            Y_adv[idx_noise, ...] = Y_adv_cur.cpu()
            Y_ref[idx_noise, ...] = Y_ref_cur.cpu()

        idx_max = X_adv_err[idx_noise, ...].argsort(descending=True)
        Y_adv_cur = Y_adv_cur[idx_max[0:keep_init], ...]
    print('att: X_adv_err.shape: ', X_adv_err.shape)

    if store_data:
        return X_adv_err, X_ref_err, X_adv, X_ref, Y_adv, Y_ref
    else:
        return X_adv_err, X_ref_err
