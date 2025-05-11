import torch
import time
from utils.Stridge import (
    U_all_compute,
    lib_poly_compute,
    clean_channel,
    clean_channel_predict,
    downsample_modified,
    Train_STRidge
)

def run_STRidge_pipeline(train_loader, device, args,
                         upper_bound_x, upper_bound_y, upper_bound_t,
                         lower_bound_x, lower_bound_y, lower_bound_t):
    """
    Perform STRidge training pipeline on a batch from train_loader.

    Args:
        train_loader (DataLoader): Loader for training data.
        device (torch.device): Computation device.
        args (Namespace): Parsed command-line arguments.
        upper_bound_x, upper_bound_y, upper_bound_t,
        lower_bound_x, lower_bound_y, lower_bound_t (float): Domain boundaries.

    Returns:
        w_true (Tensor): Learned sparse coefficients.
        elapsed_time (float): Execution time in seconds.
    """
    # Fetch and preprocess a batch
    full_data, _, _, _ = next(iter(train_loader))
    full_data = full_data.to(device)
    full_data_cleaned = clean_channel(full_data)

    # Compute full field and its time derivative
    U_all_full, U_t_full = U_all_compute(
        full_data_cleaned,
        device=device,
        upper_bound_x=upper_bound_x,
        upper_bound_y=upper_bound_y,
        upper_bound_t=upper_bound_t,
        lower_bound_t=lower_bound_t,
        lower_bound_x=lower_bound_x,
        lower_bound_y=lower_bound_y
    )

    # Build polynomial library
    U = full_data_cleaned.unsqueeze(0)
    lib_poly = lib_poly_compute(
        P=args.poly_order,
        U_all=U_all_full,
        U=U,
        channel_split=args.channel_split
    )

    # Initialize random coefficients
    lambda_w = torch.randn([lib_poly.shape[0], 1]).to(device)

    # STRidge parameters
    lam = 1e-5
    d_tol = args.tol
    maxit = 100
    normalize = 2
    l0_penalty = 1e-4

    # Train STRidge and measure time
    start_time = time.time()
    w_true, _ = Train_STRidge(
        lib_poly=lib_poly,
        U_t=U_t_full,
        device=device,
        lam=lam,
        maxit=maxit,
        normalize=normalize,
        lambda_w=lambda_w,
        l0_penalty=l0_penalty,
        print_best_tol=False,
        d_tol=d_tol,
        channel_split=args.channel_split
    )
    elapsed_time = time.time() - start_time

    return w_true, elapsed_time


def run_STRidge_prediction(xx, device, args,
                           upper_bound_x, upper_bound_y, upper_bound_t,
                           lower_bound_x, lower_bound_y, lower_bound_t,
                           P, lam=1e-5, d_tol=1e-5,
                           maxit=100, normalize=2, l0_penalty=1e-4):
    """
    Perform STRidge prediction pipeline on input data.

    Args:
        xx (Tensor): Input data tensor.
        device (torch.device): Computation device.
        args (Namespace): Parsed command-line arguments.
        upper_bound_x, upper_bound_y, upper_bound_t,
        lower_bound_x, lower_bound_y, lower_bound_t (float): Domain boundaries.
        P (int): Polynomial order.
        lam, d_tol, maxit, normalize, l0_penalty (float/int): STRidge parameters.

    Returns:
        w_best (Tensor): Estimated coefficients.
        phi_loss (float): Loss value from STRidge.
        elapsed_time (float): Execution time in seconds.
    """
    # Preprocess and downsample
    xx_cleaned = clean_channel_predict(xx)
    xx_downsampled = downsample_modified(
        xx_cleaned,
        batch=args.batch_down,
        grid_xy=args.grid_xy,
        time_hold=args.time_cut,
        device=device,
        seed=args.seed
    )

    # Compute field and time derivative
    U_all, U_t = U_all_compute(
        xx_downsampled,
        device=device,
        upper_bound_x=upper_bound_x,
        upper_bound_y=upper_bound_y,
        upper_bound_t=upper_bound_t,
        lower_bound_t=lower_bound_t,
        lower_bound_x=lower_bound_x,
        lower_bound_y=lower_bound_y
    )

    # Build polynomial library
    U = xx_downsampled.unsqueeze(0)
    lib_poly = lib_poly_compute(
        P=P,
        U_all=U_all,
        U=U,
        channel_split=args.channel_split
    )

    # Initialize random coefficients
    lambda_w = torch.randn([lib_poly.shape[0], 1]).to(device)

    # Replace NaN/Inf values if detected
    nan_inf_mask = torch.isnan(lib_poly) | torch.isinf(lib_poly)
    if nan_inf_mask.any() or torch.isnan(U).any() or torch.isinf(U).any() or torch.isnan(U_t).any() or torch.isinf(U_t).any():
        lib_poly = torch.nan_to_num(lib_poly)
        U_t = torch.nan_to_num(U_t)

    # Train STRidge and measure time
    start_time = time.time()
    w_best, phi_loss = Train_STRidge(
        lib_poly=lib_poly,
        U_t=U_t,
        device=device,
        lam=lam,
        maxit=maxit,
        normalize=normalize,
        lambda_w=lambda_w,
        l0_penalty=l0_penalty,
        print_best_tol=False,
        d_tol=d_tol,
        channel_split=args.channel_split
    )
    elapsed_time = time.time() - start_time

    return w_best, phi_loss, elapsed_time
