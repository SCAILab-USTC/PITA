import torch


def get_field_bcs(
    channel = 2, 
    batch_size = 20,
    dataset = 'dr_pdb'
):
    if dataset == 'dr_pdb':
        field_labels = torch.tensor([[4, 5]] * batch_size)
        bcs = torch.tensor([[0,0]] * batch_size)
    elif dataset == 'swe_pdb':
        field_labels = torch.tensor([[0]] * batch_size)
        bcs = torch.tensor([[0,0]] * batch_size)
    elif dataset == 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2' or dataset == 'ns2d_pdb_M1_eta1e-1_zeta1e-1' or dataset == 'ns2d_pdb_M1_eta1e-2_zeta1e-2' or dataset == 'ns2d_pdb_M1e-1_eta1e-1_zeta1e-1':
        field_labels = torch.tensor([[6,7,8,9]] * batch_size)
        bcs = torch.tensor([[1,1]]* batch_size)
    elif dataset == 'ns2d_pdb_M1_eta1e-8_zeta1e-8_turb_512':
        field_labels = torch.tensor([[6,7,8,9]] * batch_size)
        bcs = torch.tensor([[1,1]]* batch_size)       
        
    return field_labels, bcs




def get_grid_bound_2D(dataset, T_ar = 10):
    upper_bound_x, upper_bound_y, upper_bound_t, lower_bound_x, lower_bound_y, lower_bound_t = 0,0,0,0,0,0
    
    if dataset == "swe_pdb":
        upper_bound_x = 2.5
        upper_bound_y = 2.5 
        upper_bound_t = 1  
        lower_bound_x = -2.5
        lower_bound_y = -2.5
        lower_bound_t = 0
    elif dataset == "dr_pdb":
        upper_bound_x = 1
        upper_bound_y = 1
        upper_bound_t = 5
        lower_bound_x = -1
        lower_bound_y = -1
        lower_bound_t = 0       
    elif dataset[:10]=="ns2d_fno_1":
        upper_bound_x = 1
        upper_bound_y = 1
        upper_bound_t = T_ar
        lower_bound_x = 0
        lower_bound_y = 0
        lower_bound_t = 0  
    elif dataset[:10] == 'ns2d_pdb_M':
        upper_bound_x = 1
        upper_bound_y = 1
        upper_bound_t = 1
        lower_bound_x = 0
        lower_bound_y = 0
        lower_bound_t = 0  
    elif dataset == 'cfdbench':
        upper_bound_x = 1
        upper_bound_y = 1
        upper_bound_t = 1
        lower_bound_x = 0
        lower_bound_y = 0
        lower_bound_t = 0  
    elif dataset == 'ns2d_pda' or dataset == 'sw2d_pda' or dataset == 'ns2d_cond_pda':
        upper_bound_x = 32
        upper_bound_y = 32
        upper_bound_t = 24
        lower_bound_x = 0
        lower_bound_y = 0
        lower_bound_t = 0      
    elif dataset == 'burgers':
        upper_bound_x = 64.25
        upper_bound_y = 64.25
        upper_bound_t = 9.8
        lower_bound_x = 0.25
        lower_bound_y = 0.25
        lower_bound_t = 0
    return  upper_bound_x, upper_bound_y, upper_bound_t, lower_bound_x, lower_bound_y, lower_bound_t