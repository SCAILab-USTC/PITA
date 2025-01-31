Codes for ICML 2025 Paper "Physics-informed Temporal Alignment for Auto-regressive PDE Foundation Models".

## Requirements
To run the experiments successfully, you need to prepare the envirment.
```
conda env create -f environment.yaml
```

## Training
- For DPOT:
```
python train_PITA_DPOT.py --gpu 7 --epochs 500 --time_cut 3 --resume_path models/DPOT/model_M.pth --model_size M --train_paths ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512 --test_paths ns2d_pdb_M1e-1_eta1e-8_zeta1e-8_turb_512 --ntrain_list 900 --lr_method cycle 
```
- For FNO:
```
python train_PITA_FNO.py --gpu 6 --train_paths ns2d_pdb_M1_eta1e-2_zeta1e-2 --test_paths ns2d_pdb_M1_eta1e-2_zeta1e-2 --ntrain_list 9000 --lr_method cycle 


- For MPP:

```
python train_MPP.py --gpu 1 --model_size Ti --train_paths burgers --test_paths burgers --ntrain_list 1000 --lr_method cycle --T_in 10 --T_ar 1 --lr 0.001
```
