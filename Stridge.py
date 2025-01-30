
import torch
import numpy as np



def STRidge( X0, y, lam, maxit, tol, normalize=2, print_results = False, device = "cuda:0", lambda_w = torch.zeros(60,1) ):

    n, d = X0.shape
    X = torch.zeros((n, d), dtype=y.dtype).to(device)
    y = y.to(device)
        
    # Convert X0 to PyTorch tensor
    # X0 = torch.tensor(X0, dtype=torch.float32,device=device)  # Use appropriate dtype based on your data
    ############ Version 1
    # First normalize data
    # if normalize != 0:
    #     Mreg = torch.zeros((d, 1),device=device)
    #     for i in range(d):
    #         Mreg[i] = 1.0 / (torch.norm(X0[:, i], normalize))
    #         X[:, i] = Mreg[i] * X0[:, i]
    # else:
    #     X = X0
    ############ Version 2
    # if normalize != 0:
    #     Mreg = torch.zeros((d, 1), device=device)
    #     X_list = []
    #     for i in range(d):
    #         Mreg[i] = 1.0 / (torch.norm(X0[:, i], normalize))
    #         X_list.append(Mreg[i] * X0[:, i].unsqueeze(1))          # else:
    #     X = X0
    if normalize != 0:
        Mreg = torch.zeros((d, 1), device=device)  
        X_list = []  
        for i in range(d):
            Mreg[i] = 1.0 / (torch.norm(X0[:, i], normalize))  
            normalized_col = Mreg[i] * X0[:, i].unsqueeze(1)  
            X_list.append(normalized_col.detach().clone())  
        X = torch.cat(X_list, dim=1)  
    else:
        X = X0.clone()  
    epsilon = 1e-8  
    w = lambda_w.clone() / (Mreg + epsilon)

    # Inherit w from previous training
    # w = lambda_w.clone() / Mreg
    # w = lambda_w / Mreg

    num_relevant = d
    biginds = torch.where(abs(w) > tol)[0].to(device)

    # ridge_append_counter = 0

    lambda_history_STRidge = []
    lambda_history_STRidge.append(Mreg * w)
    # lambda_history_STRidge  = Mreg * w
    # ridge_append_counter += 1

    # Threshold and continue
    for j in range(maxit):
        # Figure out which items to cut out
        smallinds = torch.where(abs(w) < tol)[0].to(device)
        new_biginds = [i for i in range(d) if i not in smallinds.cpu().numpy()]  # Convert to CPU for list operations
    
        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                if normalize != 0:
                    w = Mreg * w
                    lambda_history_STRidge.append(w)
                    # lambda_history_STRidge = torch.cat((lambda_history_STRidge,w))
                    # ridge_append_counter += 1
                    return w
                else:
                    lambda_history_STRidge.append(w)
                    # lambda_history_STRidge = torch.cat((lambda_history_STRidge,w))
                    # ridge_append_counter += 1
                    return w
            else:
                break
        biginds = torch.tensor(new_biginds).to(device)

        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = torch.real(torch.linalg.lstsq(X[:, biginds].T @ X[:, biginds] + torch.tensor(lam).to(device) * torch.eye(len(biginds), dtype=X.dtype,device=device), 
                                                        X[:, biginds].T @ y, rcond=None).solution)
            lambda_history_STRidge.append(Mreg * w)
            # lambda_history_STRidge = torch.cat((lambda_history_STRidge,Mreg * w))
            # ridge_append_counter += 1
        else:
            w[biginds] = torch.real(torch.linalg.lstsq(X[:, biginds], y, rcond=None).solution)
            lambda_history_STRidge.append(w)
            # lambda_history_STRidge = torch.cat((lambda_history_STRidge,Mreg * w))
            # ridge_append_counter += 1
        

    # Now that we have the sparsity pattern, use standard least squares to get w
    if len(biginds) != 0:
        #   X[:, biginds].cuda().T.cuda() @ X[:, biginds].cuda() + torch.tensor(lam).cuda() * torch.eye(len(biginds), dtype=X.dtype,device=device)

        A = X[:, biginds].T @ X[:, biginds] + torch.tensor(lam,dtype=X.dtype).to(device) * torch.eye(len(biginds), dtype=X.dtype,device=device)
        B = X[:, biginds].T @ y
        w[biginds] = torch.real(torch.linalg.lstsq(A, B, rcond=None).solution)
        
    if normalize != 0:
        w = Mreg * w
        lambda_history_STRidge.append(w)
        # lambda_history_STRidge = torch.cat((lambda_history_STRidge,w))
        # ridge_append_counter += 1
        return w
    else:
        lambda_history_STRidge.append(w)
        # lambda_history_STRidge = torch.cat((lambda_history_STRidge,w))
        # ridge_append_counter += 1
        return w
    




def build_library( data, derivatives, derivatives_description, PolyOrder=2, data_description=None):
    ## polynomial terms
    P = PolyOrder
    lib_poly = [torch.ones_like(data[0])]
    lib_poly_descr = ['']  # it denotes '1'
    for i in range(len(data)):  # polynomial terms of univariable
        for j in range(1, P + 1):
            lib_poly.append(data[i] ** j)
            lib_poly_descr.append(data_description[i] + "**" + str(j))

    lib_poly.append(data[0] * data[1])
    lib_poly_descr.append(data_description[0] + data_description[1])
    lib_poly.append(data[0] * data[2])
    lib_poly_descr.append(data_description[0] + data_description[2])
    lib_poly.append(data[1] * data[2])
    lib_poly_descr.append(data_description[1] + data_description[2])

    ## derivative terms
    lib_deri = derivatives
    lib_deri_descr = derivatives_description

    ## Multiplication of derivatives and polynomials (including the multiplication with '1')
    lib_poly_deri = []
    lib_poly_deri_descr = []
    for i in range(len(lib_poly)):
        for j in range(len(lib_deri)):
            lib_poly_deri.append(lib_poly[i] * lib_deri[j])
            lib_poly_deri_descr.append(lib_poly_descr[i] + lib_deri_descr[j])

    return lib_poly_deri, lib_poly_deri_descr



def U_all_compute(
          yy = torch.randn([20,128,128,10,3]) ,
          device = "cuda:0",
          upper_bound_x = 2.5,
          upper_bound_y = 2.5,
          upper_bound_t = 1,
          
          lower_bound_t = 0,          
          lower_bound_x = -2.5,
          lower_bound_y = -2.5
          ):
    # print(f"U_all_compute yy.shape = {yy.shape}")
    # _, size_x, size_y ,size_t= yy.shape[0], yy.shape[1], yy.shape[2],yy.shape[3]
    _, size_x, size_y ,size_t= yy.shape[0], 128, 128,yy.shape[3]
    gridx = torch.tensor(np.linspace(lower_bound_x, upper_bound_x, size_x), dtype=torch.float, device = device)
    gridy = torch.tensor(np.linspace(lower_bound_y, upper_bound_y, size_y), dtype=torch.float, device = device)   
    gridt = torch.tensor(np.linspace(lower_bound_t, upper_bound_t, size_t), dtype=torch.float, device = device)
            
    dx = gridx[1] - gridx[0]  #  step for x grid
    dy = gridy[1] - gridy[0]  # step for y grid
    dt = gridt[1] - gridt[0]  # step for t grid
    # U = yy.unsqueeze(0)
    U_t = [] #torch.zeros_like(yy)
    for i in range(yy.shape[-1]):
        yy_channel = yy[..., i:i+1]      
        du_dx = torch.gradient(yy_channel, spacing=dx, dim=1)  #: torch.Size([20, 128, 128, 10])
        du_dy = torch.gradient(yy_channel, spacing=dy, dim=2)  # y
        du_dt = torch.gradient(yy_channel, spacing=dt, dim=3)  # t 
        
        du_dx_dx = torch.gradient(du_dx[0], spacing=dx, dim=1)  # x : torch.Size([20, 128, 128, 10])
        du_dy_dx = torch.gradient(du_dy[0], spacing=dx, dim=1)  # y   
        du_dy_dy = torch.gradient(du_dy[0], spacing=dy, dim=2)  # y    
        du_dx_dy = torch.gradient(du_dx[0], spacing=dy, dim=2)  # y   
                                        

        if i==0:
            U_t ,U_x, U_y, U_xy, U_xx, U_yy, U_yx= du_dt[0],du_dx[0],du_dy[0],du_dx_dy[0],du_dx_dx[0],du_dy_dy[0],du_dy_dx[0]
        
        else:
            U_t = torch.cat((U_t, du_dt[0]), dim=-1)
            U_x = torch.cat((U_x, du_dx[0]), dim=-1)
            U_y = torch.cat((U_y, du_dy[0]), dim=-1)
            U_xy = torch.cat((U_xy, du_dx_dy[0]), dim=-1)
            U_xx = torch.cat((U_xx, du_dx_dx[0]), dim=-1)
            U_yy = torch.cat((U_yy, du_dy_dy[0]), dim=-1)
            U_yx = torch.cat((U_yx, du_dy_dx[0]), dim=-1)

    U_all = torch.stack([yy, U_x, U_y, U_xy, U_xx, U_yy, U_yx], dim=0)
    
    return U_all, U_t


def lib_poly_compute(P = 3,
             U_all = torch.randn([10,20,128,128,10,2],device = "cuda:0"),
             U = torch.randn([1,20,128,128,10,2],device = "cuda:0") 
             ):
    
    for i in range(U.shape[0]):
        for j in range(1, P+1):
            # print(f"U[i,...]**j.shape = {U[i,...]**j.shape}")
            if i ==0 and j==1:  
                lib_poly = U[i:i+1,...]**j
            else:
                lib_poly = torch.cat( (lib_poly, U[i:i+1,...]**j),  dim = 0)
                # print(f"U[i,...]**j.shape = {(U[i,...]**j).shape}")
    lib_poly = torch.cat( (lib_poly,  torch.ones_like(U[0:1, ...], device=U_all.device) ) , dim = 0)
    U_poly = lib_poly
    
    for i in range(U_all.shape[0]):
        for j in range(U_poly.shape[0]):
            lib_poly = torch.cat( (lib_poly, U_all[i:i+1,...] * U_poly[j:j+1,...]),  dim = 0)
            
    # print(f"lib_poly.shape = {lib_poly.shape}\t U_all = {U_all.shape}")
    lib_poly = torch.cat( (lib_poly, U_all), dim = 0 )
    return lib_poly

        # torch.Size([54, 20, 128, 128, 10, 2])
        # lib_poly = lib_poly_compute()
        # print(lib_poly.shape)


def Train_STRidge(lib_poly = torch.randn([16, 20, 128, 128, 10, 2], device = "cuda:0"),
                  U_t = torch.randn([1, 20, 128, 128, 10, 2], device = "cuda:0"),
                  device = "cuda:0",
                  lam = 1e-5,
                  maxit = 100,
                  normalize = 2,
                  lambda_w = torch.randn( [16,1]).to("cuda:0"),
                  l0_penalty = 1e-4,
                  print_best_tol = False ,
                  d_tol = 1
                  ):
    
    flattened_dim = torch.prod(torch.tensor(lib_poly.shape[1:])).item()  #

    lib_poly = lib_poly.view(lib_poly.shape[0], flattened_dim).transpose(0, 1) # 
    # >>> print(lib_poly.shape)
    # torch.Size([6553600, 16])
    U_t = U_t.reshape(-1,1)
    # >>> print(U_t.shape)
    # torch.Size([6553600, 1])
    
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
    
    # tol_best = 0
    # tol = d_tol
    # if l0_penalty is None:
    return w_best, err_best     


def Train_STRidge_old(lib_poly = torch.randn([16, 20, 128, 128, 10, 2], device = "cuda:0"),
                  U_t = torch.randn([1, 20, 128, 128, 10, 2], device = "cuda:0"),
                  device = "cuda:0",
                  lam = 1e-5,
                  maxit = 100,
                  normalize = 2,
                  lambda_w = torch.randn( [16,1]).to("cuda:0"),
                  l0_penalty = None,
                  print_best_tol = False ,
                  d_tol = 1
                  ):
        

    
    flattened_dim = torch.prod(torch.tensor(lib_poly.shape[1:])).item()  
   
    lib_poly = lib_poly.view(lib_poly.shape[0], flattened_dim) 
    lib_poly = lib_poly.transpose(0, 1)
    U_t = U_t.reshape(-1,1) 
    

    w_best = STRidge(X0 = lib_poly, 
                y = U_t , 
                lam = lam,  
                maxit = maxit,
                tol = d_tol,
                normalize = normalize,
                device = device,
                lambda_w = lambda_w
                )

    err_f = torch.mean(( U_t - lib_poly @ w_best.to(device)  ) ** 2)
    err_lambda = l0_penalty * torch.count_nonzero(w_best)
    err_best = err_lambda + err_f
    tol_best = 0
    tol = d_tol
    if l0_penalty is None:
        l0_penalty = err_f.item()
        
    print(f"err_f = {err_f}\t err_lambda = {err_lambda}\t err_best = {err_best}")
    
    # Now increase tolerance until test performance decreases
    loss_history_STRidge = []
    loss_f_history_STRidge = []
    loss_lambda_history_STRidge = []
    tol_history_STRidge = []

    loss_history_STRidge.append(err_best.item())
    loss_f_history_STRidge.append(err_f.item())
    loss_lambda_history_STRidge.append(err_lambda.item())
    tol_history_STRidge.append(tol_best)
    
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(X0 = lib_poly, 
            y = U_t , 
            lam = lam,  
            maxit = maxit,
            tol = d_tol,
            normalize = normalize,
            device = device,
            lambda_w = lambda_w
            )
        # err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
        err_f = torch.mean(( U_t - lib_poly @ w.to(device)  ) ** 2)
        err_lambda = l0_penalty * torch.count_nonzero(w)
        err = err_f + err_lambda

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol += d_tol

            loss_history_STRidge.append(err_best.item())
            loss_f_history_STRidge.append(err_f.item())
            loss_lambda_history_STRidge.append(err_lambda.item())
            tol_history_STRidge.append(tol)

        else:
            tol = max(0, tol - 2 * d_tol)
            d_tol = 2 * d_tol / (maxit - iter)
            tol += d_tol
    print(f"err_f = {err_f}\t err_lambda = {err_lambda}\t err_best = {err_best}")
    if print_best_tol:
        print("Optimal tolerance:", tol_best)
    return w_best        

def downsample( yy = torch.randn([20, 128, 128, 10, 3]),#.to("cuda:1"), 
               batch = 4, 
               grid_xy = 5, 
               time_cut = 2, 
               device = "cuda:1",
               seed = 42):
    


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
   

    index1 = torch.randint(0, yy.shape[0], (yy.shape[0] // batch,))
    yy_downsampled = yy[index1]


    index2 = torch.randint(0, yy.shape[1], (yy.shape[1] // grid_xy,))
    yy_downsampled = yy_downsampled[:, index2, :, :, :]
    yy_downsampled = yy_downsampled[:,:,index2,:,:]
    
    # index3 = torch.randint(0, yy.shape[3], (yy.shape[3] // time_cut,))
    yy_downsampled = yy_downsampled[:, :, :,(yy_downsampled.shape[-2]-time_hold):  , :]  

    # print(yy_downsampled.shape)  
    return yy_downsampled
