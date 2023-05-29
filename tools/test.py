import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
import os
print(f'[work_dis: {os.getcwd()}]')
import sys
sys.path.append('.')
from model import GhostNetV2P3_RFB_RFPEM as USE_MODEL
from model import CrowdDataset, CrowdDataset_p2
import numpy as np
import time
def fps(img_root, gt_dmap_root, model_param_path):
    '''
    fps
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''

    device = torch.device("cuda")
    model = USE_MODEL(mode='test', width=1.0, use_se=True, use_CAN=False, use_dcn_mode=1, gamma=True)
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    model.eval()

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    dummy_input = torch.rand(1, 3, 1024, 768).to(device)
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    # torch.cuda.synchronize()
    
    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    
    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time
            
    avg = timings.sum()/repetitions
    print('\navg={}ms\n'.format(avg))
    print('\nfps={}\n'.format(1000/avg))
    
def fps1(img_root, gt_dmap_root, model_param_path):
    '''
    fps
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''

    device = torch.device("cuda")
    model = USE_MODEL(mode='test', width=1.0, use_se=True, use_CAN=False, use_dcn_mode=1, gamma=True)
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    model.eval()

    num_warmup = 5
    pure_inf_time = 0
    fps = 0
    
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    dummy_input = torch.rand(1, 3, 1024, 768).to(device)
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    # torch.cuda.synchronize()
    
    max_iter = 2000
    log_interval = 50
    print('testing ...\n')
    with torch.no_grad():
        for i in tqdm(range(max_iter)):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            
            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(
                        f'Done image [{i + 1:<3}/ {max_iter}], '
                        f'fps: {fps:.1f} img / s, '
                        f'times per image: {1000 / fps:.1f} ms / img',
                        flush=True)

            if (i + 1) == max_iter:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Overall fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)
                break

            
    print('\nfps={}\n'.format(fps))    

def cal_mae(img_root, gt_dmap_root, model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''

    device = torch.device("cuda")
    model = USE_MODEL(width=1.0, use_se=True, use_CAN=False, use_dcn_mode=1, gamma=True)
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    dataset = CrowdDataset(img_root, gt_dmap_root, gt_downsample=8, phase='test')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    mae = 0
    with torch.no_grad():
        for i, (img, gt_dmap) in enumerate(tqdm(dataloader)):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # forward propagation
            et_dmap, et_dmap_p2 = model(img)
            mae += abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img, gt_dmap, et_dmap, et_dmap_p2
    mean_mae = mae / len(dataloader)
    print("model_param_path: {}, mae: {}".format(model_param_path, mean_mae))


def estimate_density_map(img_root, gt_dmap_root, model_param_path, index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device = torch.device("cuda")
    model = GDNet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    dataset = CrowdDataset(img_root, gt_dmap_root, 8, phase='test')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)
    model.eval()
    for i, (img, gt_dmap) in enumerate(dataloader):
        if i == index:
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # forward propagation
            et_dmap = model(img).detach()
            et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap, cmap=CM.jet)
            break


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    img_root = 'data/shanghaitech/ShanghaiTech/part_A/test_data/images'
    gt_dmap_root = 'data/shanghaitech/ShanghaiTech/part_A/test_data/ground-truth'
    model_param_path = 'checkpoints/train-GDNet-OCRFB-PFAM-RFPEM-RHCloss_ShanghaiTech_part_A/20230526_190516/train-GDNet-OCRFB-PFAM-RFPEM-RHCloss_ShanghaiTech_part_A_best_epoch_116.pth'
    # cal_mae(img_root, gt_dmap_root, model_param_path)
    fps(img_root, gt_dmap_root, model_param_path)
    # estimate_density_map(img_root,gt_dmap_root,model_param_path,3)
