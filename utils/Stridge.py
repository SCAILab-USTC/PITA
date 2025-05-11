import torch
import numpy as np
import torch


def analyze_cls(cls):
    """
    Analyze a classification tensor to find unique elements and the most common one.

    Args:
        cls (Tensor): Input tensor of shape (N,).

    Returns:
        unique_elements (list): Sorted list of unique elements.
        most_common_element (int): Element with highest frequency.
        most_common_count (int): Frequency of the most common element.
    """
    cls = cls.view(-1)
    unique_elements, counts = torch.unique(cls, return_counts=True)
    most_common_element = unique_elements[torch.argmax(counts)].item()
    most_common_count = counts.max().item() 
    return unique_elements.tolist(), most_common_element, most_common_count



def clean_channel(xx: torch.Tensor, fill_value: float = 1.0) -> torch.Tensor:
    """
    Remove channels that are entirely equal to a fill value.

    Args:
        xx (Tensor): Input tensor of shape (B, X, Y, T, C).
        fill_value (float): Value indicating an empty channel.

    Returns:
        Tensor: Cleaned tensor with invalid channels removed along last dim.
    """
    channels = xx.shape[-1]
    mask = [(xx[..., c] != fill_value).any() for c in range(channels)]
    valid = [i for i, ok in enumerate(mask) if ok]
    return xx.index_select(-1, torch.tensor(valid, device=xx.device))



def clean_channel_predict(xx: torch.Tensor, fill_value: float = 1.0) -> torch.Tensor:
    """
    Remove channels that remain equal to fill value over all but the last time step.
    """
    t = xx.shape[-2]
    channels = xx.shape[-1]
    mask = [(xx[..., :t-1, c] != fill_value).any() for c in range(channels)]
    valid = [i for i, ok in enumerate(mask) if ok]
    return xx.index_select(-1, torch.tensor(valid, device=xx.device))



def nan_inf_detect(U: torch.Tensor, U_t: torch.Tensor, lib_poly: torch.Tensor):
    """
    Detect and replace NaN or Inf values in tensors.

    Args:
        U, U_t, lib_poly (Tensor): Tensors to check.

    Returns:
        Tuple of cleaned (U, U_t, lib_poly).
    """
    def cleanup(x):
        return torch.nan_to_num(x,
                                nan=0.0,
                                posinf=torch.finfo(x.dtype).max,
                                neginf=torch.finfo(x.dtype).min)

    if any(torch.isnan(v).any() or torch.isinf(v).any() for v in (U, U_t, lib_poly)):
        lib_poly = cleanup(lib_poly)
        U_t = cleanup(U_t)
        U = cleanup(U)
    return U, U_t, lib_poly



def STRidge(X0: torch.Tensor, y: torch.Tensor, lam: float, maxit: int,
            tol: float, normalize: int = 2, device: str = "cuda:0",
            lambda_w: torch.Tensor = None) -> torch.Tensor:
    """
    Sequential threshold Ridge regression (STRidge) solver.

    Args:
        X0 (Tensor): Design matrix of shape (n, d).
        y (Tensor): Target vector of shape (n, 1).
        lam (float): Ridge penalty.
        maxit (int): Maximum iterations.
        tol (float): Threshold for pruning coefficients.
        normalize (int): Norm order for normalization (0 to disable).
        lambda_w (Tensor): Initial weights of shape (d, 1).

    Returns:
        Tensor: Sparse weight vector w of shape (d, 1).
    """
    n, d = X0.shape
    y = y.to(device)
    # Normalize columns if requested
    if normalize:
        Mreg = torch.tensor([1.0 / torch.norm(X0[:, i], normalize)
                             for i in range(d)], device=device).view(-1, 1)
        X = (X0 * Mreg.T).to(device)
    else:
        Mreg = torch.ones((d, 1), device=device)
        X = X0.to(device)

    w = (lambda_w.clone() if lambda_w is not None else torch.zeros(d, 1, device=device)) / (Mreg + 1e-8)
    biginds = torch.where(w.abs() > tol)[0]

    for _ in range(maxit):
        smallinds = torch.where(w.abs() < tol)[0]
        new_big = torch.tensor([i for i in range(d) if i not in smallinds.tolist()], device=device)
        if new_big.numel() == biginds.numel():
            break
        biginds = new_big
        w[smallinds] = 0
        if lam:
            A = X[:, biginds].T @ X[:, biginds] + lam * torch.eye(len(biginds), device=device)
            sol = torch.linalg.lstsq(A, X[:, biginds].T @ y, rcond=None).solution
        else:
            sol = torch.linalg.lstsq(X[:, biginds], y, rcond=None).solution
        w[biginds] = sol

    # Final least squares on support
    if biginds.numel() > 0:
        A = X[:, biginds].T @ X[:, biginds] + lam * torch.eye(len(biginds), device=device)
        sol = torch.linalg.lstsq(A, X[:, biginds].T @ y, rcond=None).solution
        w[biginds] = sol

    return Mreg * w if normalize else w

    
def build_library(data: list, derivatives: list, descriptions: list, order: int = 2):
    """
    Construct polynomial and derivative library for PDE terms.
    """
    lib_poly = [torch.ones_like(data[0])]
    lib_descr = ['1']
    for i, d in enumerate(data):
        for j in range(1, order + 1):
            lib_poly.append(d**j)
            lib_descr.append(f"d{i}^{j}")
    # pairwise products
    for (u, v), di in zip(zip(data, data[1:] + data[:1]), descriptions):
        lib_poly.append(u*v)
        lib_descr.append(di)
    # combine with derivatives
    combined = []
    descr_combined = []
    for p, pd in zip(lib_poly, lib_descr):
        for q, qd in zip(derivatives, descriptions):
            combined.append(p*q)
            descr_combined.append(f"{pd}_{qd}")
    return combined, descr_combined

# def build_library( data, derivatives, derivatives_description, PolyOrder=2, data_description=None):
#     ## polynomial terms
#     P = PolyOrder
#     lib_poly = [torch.ones_like(data[0])]
#     lib_poly_descr = ['']  # it denotes '1'
#     for i in range(len(data)):  # polynomial terms of univariable
#         for j in range(1, P + 1):
#             lib_poly.append(data[i] ** j)
#             lib_poly_descr.append(data_description[i] + "**" + str(j))

#     lib_poly.append(data[0] * data[1])
#     lib_poly_descr.append(data_description[0] + data_description[1])
#     lib_poly.append(data[0] * data[2])
#     lib_poly_descr.append(data_description[0] + data_description[2])
#     lib_poly.append(data[1] * data[2])
#     lib_poly_descr.append(data_description[1] + data_description[2])

#     ## derivative terms
#     lib_deri = derivatives
#     lib_deri_descr = derivatives_description

#     ## Multiplication of derivatives and polynomials (including the multiplication with '1')
#     lib_poly_deri = []
#     lib_poly_deri_descr = []
#     for i in range(len(lib_poly)):
#         for j in range(len(lib_deri)):
#             lib_poly_deri.append(lib_poly[i] * lib_deri[j])
#             lib_poly_deri_descr.append(lib_poly_descr[i] + lib_deri_descr[j])

#     return lib_poly_deri, lib_poly_deri_descr

def U_all_compute(yy: torch.Tensor, device: str,
                  upper_bound_x: float, upper_bound_y: float,
                  upper_bound_t: float, lower_bound_t: float,
                  lower_bound_x: float, lower_bound_y: float):
    """
    Compute spatial and temporal derivatives of tensor yy.

    Returns:
        U_all (Tensor): Stacked array of field and derivatives.
        U_t (Tensor): Time derivative tensor.
    """
    _, nx, ny, nt, nc = yy.shape
    dx = (upper_bound_x - lower_bound_x) / (nx - 1)
    dy = (upper_bound_y - lower_bound_y) / (ny - 1)
    dt = (upper_bound_t - lower_bound_t) / (nt - 1)

    derivatives = []
    for i in range(nc):
        u = yy[..., i:i+1]
        du_dx = torch.gradient(u, spacing=dx, dim=1)[0]
        du_dy = torch.gradient(u, spacing=dy, dim=2)[0]
        du_dt = torch.gradient(u, spacing=dt, dim=3)[0]
        derivatives.append((du_dt, du_dx, du_dy))
    U_t = torch.cat([d[0] for d in derivatives], dim=-1)
    U_all = torch.stack([yy] + [d for tup in derivatives for d in tup[1:]], dim=0)
    return U_all, U_t



def U_part_compute(yy: torch.Tensor, device: str,
                   upper_bound_x: float, upper_bound_y: float,
                   upper_bound_t: float, lower_bound_t: float,
                   lower_bound_x: float, lower_bound_y: float):
    """
    Partial derivative library without second-order terms.
    """
    U_all, U_t = U_all_compute(yy, device,
                                upper_bound_x, upper_bound_y,
                                upper_bound_t, lower_bound_t,
                                lower_bound_x, lower_bound_y)
    # Drop second-order derivatives
    return U_all[:4], U_t



def lib_poly_compute(P = 3,
             U_all = torch.randn([10,20,128,128,10,2],device = "cuda:0"),
             U = torch.randn([1,20,128,128,10,2],device = "cuda:0"), 
             channel_split = False
             ):
    """
    Build a library of polynomial and interaction terms.

    Args:
        P: Maximum polynomial power.
        U_all: Tensor of field and derivatives [terms, B, X, Y, T, C].
        U: Tensor of field snapshots [1, B, X, Y, T, C].
        channel_split: Whether to cross channels separately.

    Returns:
        lib: Concatenated tensor of polynomial features.
    """
    if channel_split == False:
        for i in range(U.shape[0]):
            for j in range(1, P+1):
                if i ==0 and j==1: 
                    lib_poly = U[i:i+1,...]**j
                else:
                    lib_poly = torch.cat( (lib_poly, U[i:i+1,...]**j),  dim = 0)
        lib_poly = torch.cat( (lib_poly,  torch.ones_like(U[0:1, ...], device=U_all.device) ) , dim = 0)
        U_poly = lib_poly
        for i in range(U_all.shape[0]):
            for j in range(U_poly.shape[0]):
                lib_poly = torch.cat( (lib_poly, U_all[i:i+1,...] * U_poly[j:j+1,...]),  dim = 0)
        lib_poly = torch.cat( (lib_poly, U_all), dim = 0 )
        return lib_poly

    elif channel_split == True:
        channels = U_all.shape[-1]
        for i in range(U.shape[0]):
            for j in range(1, P+1):
                if i ==0 and j==1:  
                    lib_poly_ = U[i:i+1,...]**j
                else:
                    lib_poly_ = torch.cat( (lib_poly_, U[i:i+1,...]**j),  dim = 0)
        lib_poly_ = torch.cat( (lib_poly_,  torch.ones_like(U[0:1, ...], device=U_all.device) ) , dim = 0)
        U_poly = lib_poly_
        U_split = torch.split(U_poly, 1, dim = -1)
        U_reshaped = torch.cat( U_split, dim = 0 )
        lib_poly = U_reshaped
        
        for channel_poly in range(channels):
            for channel_u_all in range(channels):
                # for p in range(P):        
                for i in range(U_all.shape[0]):
                    for j in range(U_poly.shape[0]):
                        lib_poly = torch.cat( (lib_poly, U_all[i:i+1,...,channel_u_all:channel_u_all+1] * U_poly[j:j+1, ...,channel_poly:channel_poly+1 ] ), dim = 0)     
        return lib_poly
    
    
def Train_STRidge(lib_poly = torch.randn([16, 20, 128, 128, 10, 2], device = "cuda:0"),
                  U_t = torch.randn([1, 20, 128, 128, 10, 2], device = "cuda:0"),
                  device = "cuda:0",
                  lam = 1e-5,
                  maxit = 100,
                  normalize = 2,
                  lambda_w = torch.randn( [16,1]).to("cuda:0"),
                  l0_penalty = 1e-4,
                  print_best_tol = False ,
                  d_tol = 1,
                  channel_split = False
                  ):
    """
    Wrapper to reshape data and apply STRidge solver.

    Returns:
        w_best: Best-fit coefficients.
        err: Composite error measure.
    """
    if channel_split == False:    
        flattened_dim = torch.prod(torch.tensor(lib_poly.shape[1:])).item()  
        lib_poly = lib_poly.view(lib_poly.shape[0], flattened_dim).transpose(0, 1).contiguous() 
        U_t = U_t.reshape(-1,1)
        w_best = STRidge(
                    X0 = lib_poly, 
                    y = U_t , 
                    lam = lam,  
                    maxit = maxit,
                    tol = d_tol,
                    normalize = normalize,
                    device = device,
                    lambda_w = lambda_w
                    ) 
        
        # myloss = SimpleLpLoss(size_average=False)
        err_f = torch.mean(( U_t - lib_poly @ w_best.to(device)  ) ** 2)
        err_best = err_f + err_f.item() * torch.count_nonzero(w_best) 

        return w_best, err_best     
    
    elif channel_split == True:
        channels = U_t.shape[-1]
        err_f, err_best = 0.0, 0.0
        w_best_channels = []
        flattened_dim = torch.prod(torch.tensor(lib_poly.shape[1:])).item()  
        lib_poly = lib_poly.view(lib_poly.shape[0], flattened_dim).transpose(0, 1).contiguous() 
        for i in range(channels):
            U_t_channel = U_t[..., i:i+1]
            U_t_channel = U_t_channel.reshape(-1,1)
            w_best = STRidge(
                        X0 = lib_poly, 
                        y = U_t_channel , 
                        lam = lam,  
                        maxit = maxit,
                        tol = d_tol,
                        normalize = normalize,
                        device = device,
                        lambda_w = lambda_w
                        )
            err_i = torch.mean(( U_t_channel - lib_poly @ w_best.to(device)  ) ** 2)
            err_best += ( err_i + err_i.item() * torch.count_nonzero(w_best) )
            err_f += err_i
            w_best_channels.append(w_best)

        return torch.cat(w_best_channels, dim=1), err_best     


def downsample( yy = torch.randn([20, 128, 128, 10, 3]),#.to("cuda:1"), 
               batch = 4, 
               grid_xy = 5, 
               time_cut = 2, 
               device = "cuda:1",
               seed = 42):
    """
    Random spatial-temporal downsampling.
    """
    index1 = torch.randint(0, yy.shape[0], (yy.shape[0] // batch,))
    yy_downsampled = yy[index1]
    index2 = torch.randint(0, yy.shape[1], (yy.shape[1] // grid_xy,))
    yy_downsampled = yy_downsampled[:, index2, :, :, :]
    yy_downsampled = yy_downsampled[:,:,index2,:,:]
    index3 = torch.randint(0, yy.shape[3], (yy.shape[3] // time_cut,))
    yy_downsampled = yy_downsampled[:, :, :, index3, :]

    return yy_downsampled


def downsample_modified( yy = torch.randn([20, 128, 128, 10, 3]),#.to("cuda:1"), 
               batch = 4, 
               seed = 42, 
               grid_xy = 5, 
               time_hold = 5, 
               device = "cuda:1"):
    """
    Random spatial-temporal downsampling.
    """
    index1 = torch.randint(0, yy.shape[0], (yy.shape[0] // batch,))
    yy_downsampled = yy[index1]
    index2 = torch.randint(0, yy.shape[1], (yy.shape[1] // grid_xy,))
    yy_downsampled = yy_downsampled[:, index2, :, :, :]
    yy_downsampled = yy_downsampled[:,:,index2,:,:]
    yy_downsampled = yy_downsampled[:, :, :,(yy_downsampled.shape[-2]-time_hold):  , :]  
    return yy_downsampled
