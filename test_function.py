import torch
import os
from utils.criterion import SimpleLpLoss


myloss = SimpleLpLoss(size_average=False)

def evaluate_model(model, test_loaders, ntests, test_paths, ep, log_path, args, writer=None, save_tag=""):
    model.eval()
    test_l2_steps, test_l2_fulls = [], []

    with torch.no_grad():
        for id, test_loader in enumerate(test_loaders):
            step_loss_total, full_loss_total = 0, 0

            for xx, yy, msk, _ in test_loader:
                xx, yy, msk = xx.to(args.device), yy.to(args.device), msk.to(args.device)
                loss, t = 0, 0
                xx_input = xx.clone()

                while t < yy.shape[-2]:
                    y_target = yy[..., t:t + args.T_bundle, :]
                    y_pred, _ = model(xx_input)
                    loss += myloss(y_pred, y_target, mask=msk)

                    xx_input = torch.cat((xx_input[..., args.T_bundle:, :], y_pred), dim=-2)
                    pred = y_pred if t == 0 else torch.cat((pred, y_pred), dim=-2)
                    t += args.T_bundle

                step_loss_total += loss.item()
                full_loss_total += myloss(pred, yy, mask=msk)

                if (ep + 1) % 500 == 0:
                    save_dir = os.path.join(log_path, args.train_paths[0], "save", save_tag)
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({'ground_truth': yy, 'prediction': pred}, f'{save_dir}/test_{ep}.pt')
                    print(f"[{save_tag}] Saved prediction to {save_dir}/test_{ep}.pt")

            avg_step_loss = step_loss_total / ntests[id] / (yy.shape[-2] / args.T_bundle)
            avg_full_loss = full_loss_total / ntests[id]
            test_l2_steps.append(avg_step_loss)
            test_l2_fulls.append(avg_full_loss)

            if args.use_writer and writer:
                writer.add_scalar(f"test_loss_step_{test_paths[id]}", avg_step_loss, ep)
                writer.add_scalar(f"test_loss_full_{test_paths[id]}", avg_full_loss, ep)

    return test_l2_fulls, test_l2_steps
