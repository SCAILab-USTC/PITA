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



def get_grid_bound_2D(dataset: str, T_ar: int = 10) -> tuple:
    """
    Return spatial and temporal bounds for a 2D dataset.

    Args:
        dataset: Name of the dataset.
        T_ar: Time horizon for datasets with variable temporal bounds.

    Returns:
        (upper_x, upper_y, upper_t, lower_x, lower_y, lower_t)
    """
    # Predefined bounds per dataset
    bounds = {
        "swe_pdb":      (2.5, 2.5, 1, -2.5, -2.5, 0),
        "dr_pdb":       (1, 1, 5, -1, -1, 0),
        "ns2d_fno_1":   (1, 1, T_ar, 0, 0, 0),
        "ns2d_fno_500": (1, 1, T_ar, 0, 0, 0),
        "ns2d_pdb_M":   (1, 1, 1, 0, 0, 0),
        "cfdbench":     (1, 1, 1, 0, 0, 0),
        "pda_group":    (32, 32, 24, 0, 0, 0),
        "burgers":      (64.25, 64.25, 9.8, 0.25, 0.25, 0),
    }
    # Identify PDA group datasets
    if dataset in ("ns2d_pda", "sw2d_pda", "ns2d_cond_pda"):
        key = "pda_group"
    elif dataset.startswith("ns2d_fno_1"):
        key = "ns2d_fno_1"
    elif dataset.startswith("ns2d_pdb_M"):
        key = "ns2d_pdb_M"
    else:
        key = dataset

    return bounds.get(key, (0, 0, 0, 0, 0, 0))
