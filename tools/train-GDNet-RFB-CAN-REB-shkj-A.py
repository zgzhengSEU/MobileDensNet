import time
import torch
import torch.nn as nn
import os
from tqdm import tqdm as tqdm
import time
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast as autocast
import random
from matplotlib import pyplot as plt
from torchvision import transforms
import cv2

import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_warmup as warmup
from torch.cuda.amp import GradScaler
import wandb
from collections import OrderedDict
import math
from torch.utils.data import DataLoader
print(f'[work_dis: {os.getcwd()}]')
import sys
sys.path.append('.')

from model import GhostNetV2P3_RFB_CAN_REB as USE_MODEL
from model import CrowdDataset, CrowdDataset_p2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--show_images', type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_id', type=str, default='5tpdfo8k')
    parser.add_argument('--resume_checkpoint', type=str, default='')
    parser.add_argument('--init_checkpoint', type=str, default='checkpoints/ghostnetv2_torch/ck_ghostnetv2_16.pth.tar')
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    return args

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # ===================== DataPath =========================
    datatype = 'ShanghaiTech_part_A'
    # datatype = 'ShanghaiTech_part_B'
    # datatype = 'VisDrone2020-CC'
    # datatype = 'VisDrone'
    if datatype == 'ShanghaiTech_part_A':
        train_image_root = 'data/shanghaitech/ShanghaiTech/part_A/train_data/images'
        train_dmap_root = 'data/shanghaitech/ShanghaiTech/part_A/train_data/ground-truth'
        test_image_root = 'data/shanghaitech/ShanghaiTech/part_A/test_data/images'
        test_dmap_root = 'data/shanghaitech/ShanghaiTech/part_A/test_data/ground-truth'
    elif datatype == 'ShanghaiTech_part_B':
        train_image_root = 'data/shanghaitech/ShanghaiTech/part_B/train_data/images'
        train_dmap_root = 'data/shanghaitech/ShanghaiTech/part_B/train_data/ground-truth'
        test_image_root = 'data/shanghaitech/ShanghaiTech/part_B/test_data/images'
        test_dmap_root = 'data/shanghaitech/ShanghaiTech/part_B/test_data/ground-truth'
    elif datatype == 'VisDrone2020-CC':
        train_image_root = 'data/VisDrone2020-CC/train/images'
        train_dmap_root = 'data/VisDrone2020-CC/train/ground_truth'
        test_image_root = 'data/VisDrone2020-CC/test/images'
        test_dmap_root = 'data/VisDrone2020-CC/test/ground_truth'
    elif datatype == 'VisDrone':
        train_image_root = 'data/DensVisDrone/images/train'
        train_dmap_root = 'data/DensVisDrone/density/dens_train'
        test_image_root = 'data/DensVisDrone/images/val'
        test_dmap_root = 'data/DensVisDrone/density/dens_val'
    # ===================== wandb =============================
    wandb_project="Density"
    wandb_group=datatype
    wandb_mode="online"
    wandb_name='GhostNetV2P3_RFB_CAN_REB'
    # ===================== configuration ======================
    init_checkpoint = args.init_checkpoint
    temp_init_checkpoint_path = "checkpoints"
    resume_checkpoint = args.resume_checkpoint
    use_amp = args.amp
    print(f'use_amp: {use_amp}')
    print(f'use_wandb: {args.wandb}')
    use_wandb = args.wandb
    show_images = args.show_images
    resume = args.resume
    resume_id = args.resume_id
    
    lr = args.lr
    gpu_or_cpu = args.device  # use cuda or cpu

    start_epoch = 0
    epochs = args.epochs
    train_num_workers = 4
    test_num_workers = 4
    seed = time.time()
    import datetime as dt
    NOW_TIME = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = f'./checkpoints/{wandb_name}_{datatype}/{NOW_TIME}'

    # ==========================================================
    curtime = time.strftime(
        '%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
    print(f"[{curtime}] GPU train start!")
    print(args)

    if os.path.exists(temp_init_checkpoint_path) is False:
        os.makedirs(temp_init_checkpoint_path)
    if os.path.exists('checkpoints/temp/') is False:
        os.makedirs('checkpoints/temp/')
    if os.path.exists(work_dir) is False:
        os.makedirs(work_dir)
    if use_wandb:
        if resume:
            wandb.init(
                project=wandb_project,
                group=wandb_group,
                mode=wandb_mode,
                resume='allow',
                id = resume_id,
                name=wandb_name)
        else:
            wandb.init(
                project=wandb_project,
                group=wandb_group,
                mode=wandb_mode,
                name=wandb_name,
                settings=wandb.Settings(code_dir="."))

    # ======================== cuda ===================================================================
    device = torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    # ======================== dataloader =================================================================
    train_dataset = CrowdDataset_p2(train_image_root, train_dmap_root, gt_downsample=8, phase='train')
    test_dataset = CrowdDataset(test_image_root, test_dmap_root, gt_downsample=8, phase='test')
    train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True, num_workers=train_num_workers, pin_memory=True)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False, num_workers=test_num_workers, pin_memory=True)
    # ========================================= model =================================================
    model = USE_MODEL(width=1.6, use_se=True, use_CAN=True).to(device)

    if resume:
        resume_load_checkpoint = torch.load(resume_checkpoint, map_location=device)
        start_epoch = resume_load_checkpoint['epoch']
        model.load_state_dict(resume_load_checkpoint['model_state_dict'])
        # ========================= optimizer ========================================================
        pg = [p for p in model.parameters() if p.requires_grad]
        num_steps = len(train_loader) * epochs
        optimizer = optim.AdamW(pg, lr=lr, betas=(
            0.9, 0.999), weight_decay=1e-4)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        # warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
        if use_amp:
            scaler = GradScaler()
            scaler.load_state_dict(resume_load_checkpoint["scaler"])
        else:
            scaler = None    
        optimizer.load_state_dict(resume_load_checkpoint['optim_state_dict'])
        # scheduler.load_state_dict(resume_load_checkpoint['scheduler'])
        # warmup_scheduler.load_state_dict(resume_load_checkpoint['warmup_scheduler'])
    elif os.path.exists(init_checkpoint):
        # weights_dict = torch.load(init_checkpoint, map_location=device)
        # model.load_state_dict(weights_dict, strict=False)
        incompatible_keys = load_checkpoint(model, init_checkpoint, strict=False, map_location=device)
        incompatible_keys = [key for key in incompatible_keys[0] if 'total' not in key]
        # load_checkpoint = torch.load(init_checkpoint)
        # model.load_state_dict(load_checkpoint['model'].state_dict(), strict=False)
        
        print(f'[load checkpoint from {init_checkpoint}]')
        print(f'incompatible_keys:\n {incompatible_keys}')
    # ===================================== optimizer ===========================================
    if not resume:
        pg = [p for p in model.parameters() if p.requires_grad]
        # num_steps = len(train_loader) * epochs
        # optimizer = optim.AdamW(pg, lr=lr)
        optimizer = optim.SGD(pg, lr=lr, momentum=0.95, weight_decay=5e-4)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        # warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
        if use_amp:
            scaler = GradScaler()
        else:
            scaler = None
    # ========================================= train and eval ============================================
    min_mae = 1e10
    min_mse = 1e10
    min_epoch = 0
    for epoch in range(start_epoch, epochs):
        # training phase
        mean_loss = train_one_epoch_single_gpu_p2loc(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=scheduler,
            # warmup_scheduler=warmup_scheduler,
            use_amp=use_amp,
            scaler=scaler
        )
        # testing phase
        mae_sum, mse_sum = evaluate_single_gpu_p2loc(
            model=model,
            test_loader=test_loader,
            device=device,
            epoch=epoch,
            show_images=show_images,
            use_wandb=use_wandb
        )
        # eval and log
        mean_mae = mae_sum / len(test_loader)
        mean_mse = math.sqrt(mse_sum / len(test_loader))
        # checkpoints
        if os.path.exists(f'{work_dir}/{wandb_name}_{datatype}_epoch_{epoch - 1}.pth.tar') is True:
            os.remove(f'{work_dir}/{wandb_name}_{datatype}_epoch_{epoch - 1}.pth.tar')

        checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
        if use_amp:
            checkpoint_dict['scaler'] = scaler.state_dict()
            
        torch.save(checkpoint_dict, f'{work_dir}/{wandb_name}_{datatype}_epoch_{epoch}.pth.tar')

        if mean_mae < min_mae:
            min_mae = mean_mae
            min_epoch = epoch
            torch.save(checkpoint_dict['model_state_dict'],
                        f'{work_dir}/{wandb_name}_{datatype}_best_epoch_{epoch}.pth')
            torch.save(checkpoint_dict,
                        f'{work_dir}/{wandb_name}_{datatype}_best_epoch_{epoch}.pth.tar')
        
        if mean_mse < min_mse:
            min_mse = mean_mse
        
        print(
            f"[epoch {epoch}] mae: {mean_mae}, min_mae: {min_mae}, min_mse: {min_mse}, best_epoch: {min_epoch}")

        if use_wandb:
            wandb.log({'MSELoss': mean_loss})
            wandb.log({'MAE': min(mean_mae, 300)})
            wandb.log({'MSE': min(mean_mse, 300)})
            wandb.log({'MinMAE': min_mae})
            wandb.log({'MinMSE': min_mse})
            wandb.log({'MinEpoch': min_epoch})
            wandb.log({'lr': optimizer.param_groups[0]["lr"]})
            print(f"[epoch {epoch}] wandb log done!")
            
# ======================================================================================================================
def train_one_epoch_single_gpu_p2loc(model,
                    optimizer,
                    train_loader,
                    device,
                    epoch,
                    use_amp=False,
                    scaler=None,
                    lr_scheduler=None,
                    warmup_scheduler=None):
    model.train()

    criterion = nn.MSELoss(reduction='sum').to(device)

    mean_loss = torch.zeros(1).to(device)

    # 打印训练进度
    train_loader = tqdm(train_loader, file=sys.stdout)
    iters = len(train_loader)
    for step, (img, gt_dmap, gt_dmap_p2) in enumerate(train_loader):
        optimizer.zero_grad()
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        gt_dmap_p2 = gt_dmap_p2.to(device)
        if use_amp:
            # forward propagation
            with autocast():
                et_dmap_p3, et_dmap_p2 = model(img)
                # calculate loss
                # with torch.cuda.amp.autocast(enabled=False):
                et_dmap_p2 = F.interpolate(et_dmap_p2, size=(gt_dmap_p2.shape[2], gt_dmap_p2.shape[3]), mode='bilinear', align_corners=True)
                loss = criterion(et_dmap_p3, gt_dmap) + 0.0001 * criterion(et_dmap_p2, gt_dmap_p2)
            scaler.scale(loss).backward()
        else:
            # forward propagation
            et_dmap_p3, et_dmap_p2 = model(img)
            # calculate loss
            et_dmap_p2 = F.interpolate(et_dmap_p2, size=(gt_dmap_p2.shape[2], gt_dmap_p2.shape[3]), mode='bilinear', align_corners=True)
            loss = criterion(et_dmap_p3, gt_dmap) + 0.0001 * criterion(et_dmap_p2, gt_dmap_p2)
            loss.backward()
        # update mean losses
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        
        # 打印loss
        train_loader.desc = "[epoch {}] mean loss {}".format(
            epoch, round(mean_loss.item(), 3))
        
        if use_amp:
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            
            if not skip_lr_sched:
                if warmup_scheduler is not None:
                    with warmup_scheduler.dampening():
                        lr_scheduler.step(epoch + step / iters)
                else:
                    lr_scheduler.step(epoch + step / iters)
        else:
            optimizer.step()
            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    lr_scheduler.step()
            else:
                lr_scheduler.step(epoch + step / iters)
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training !!!', loss)
            sys.exit(1)
             
    return mean_loss.item()

@torch.no_grad()
def evaluate_single_gpu_p2loc(model,
             test_loader,
             device,
             epoch,
             show_images=False,
             use_wandb=False):
    model.eval()

    mae = torch.zeros(1).to(device)
    mse = torch.zeros(1).to(device)
    # 在进程中打印验证进度

    test_loader = tqdm(test_loader, file=sys.stdout)

    index = random.randint(0, len(test_loader)-1)

    for step, (img, gt_dmap) in enumerate(test_loader):
        img = img.to(device)
        gt_dmap = gt_dmap.to(device)
        # forward propagation
        et_dmap, et_dmap_p2 = model(img)
        diff = abs(et_dmap.data.sum() - gt_dmap.data.sum())
        mae += diff
        mse += diff * diff
        test_loader.desc = f"[epoch {epoch}]"

        if step == index and show_images:
            images = []
            # ============= img ==========
            _, h, w = img[0].shape
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                std=[1/0.229, 1/0.224, 1/0.255]
            )
            img_np = inv_normalize(img[0]).permute(1, 2, 0).cpu().numpy()
            # ============= gt ===========
            fig1 = plt.figure()
            plt.axis('off')
            plt.imshow(img_np)
            gt_dmap_np = gt_dmap[0].permute(1, 2, 0).cpu().numpy()
            gt_dmap_np = cv2.resize(gt_dmap_np, dsize=(w, h))
            plt.imshow(gt_dmap_np, alpha=0.5, cmap='turbo')
            fig1.savefig(f"checkpoints/temp/temp_gt_{epoch}.png",
                            bbox_inches='tight', pad_inches=0)
            fig1.clear()
            plt.close('all')
            # ============ et ==============
            fig2 = plt.figure()
            plt.axis('off')
            plt.imshow(img_np)
            et_dmap_np = et_dmap[0].permute(1, 2, 0).cpu().numpy()
            et_dmap_np = cv2.resize(et_dmap_np, dsize=(w, h))
            plt.imshow(et_dmap_np, alpha=0.5, cmap='turbo')
            fig2.savefig(f"checkpoints/temp/temp_et_{epoch}.png",
                            bbox_inches='tight', pad_inches=0)
            fig2.clear()
            plt.close('all')
            # =========== upload ============
            if use_wandb:
                images.append(wandb.Image(
                    img_np, caption=f"image {epoch}"))
                images.append(wandb.Image(plt.imread(
                    f"checkpoints/temp/temp_gt_{epoch}.png"), caption=f"gt_density {epoch}"))
                images.append(wandb.Image(plt.imread(
                    f"checkpoints/temp/temp_et_{epoch}.png"), caption=f"et_density {epoch}"))
                wandb.log({"examples": [wandb.Image(im) for im in images]})
                print(f"[epoch {epoch}] wandb upload img done!")

        del img, gt_dmap, et_dmap, et_dmap_p2

    return mae.item(), mse.item()

# ======================================================================================================================
def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict   

def load_state_dict(checkpoint_path, map_location, use_ema=True):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_checkpoint(model, checkpoint_path, map_location='cpu', use_ema=True, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, map_location, use_ema)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

if __name__ == "__main__":
    args = parse_args()
    main(args)
