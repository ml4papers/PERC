import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import sys
import torch
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
import warnings
from tqdm import tqdm
from utilities import accuracy, seed_everything
from collections import defaultdict
from TTA import ABPEM
from TTA.ABPEM import evaluate_modality_detection
# from TTA.ABPEM import READ, configure_model, collect_params, Cache, detect_corrupted_modality, unfreeze_modality_params, forward_and_adapt
from TTA.ABPEM import READ, configure_model, collect_params, Cache, forward_and_adapt



# =========================
# Args
# =========================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='vggsound', choices=['vggsound', 'ks50'])
parser.add_argument("--json-root", type=str, default='/home/jonaslei/DuBaTTA/json_csv_files/vgg')
parser.add_argument("--label-csv", type=str, default='/home/jonaslei/DuBaTTA/json_csv_files/class_labels_indices_vgg.csv')
parser.add_argument("--n_class", type=int, default=50)
parser.add_argument("--model", type=str, default='cav-mae-ft')
parser.add_argument("--dataset_mean", type=float, default=-5.081)
parser.add_argument("--dataset_std", type=float, default=4.4849)
parser.add_argument("--target_length", type=int, default=1024)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--wd', '--weight-decay', dest='wd', default=0.0, type=float)
parser.add_argument("--optim", type=str, default="adam", choices=["sgd", "adam"])
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--num-workers', default=32, type=int)
parser.add_argument("--pretrain_path", type=str, default='/home/jonaslei/DuBaTTA/data_path/vgg_65.5.pth')
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--testmode", type=str, default='multimodal')
parser.add_argument('--tta-method', type=str, default='ABPEM', choices=['ABPEM', 'None'])
parser.add_argument('--corruption-modality', type=str, default='video', choices=['video', 'audio', 'none'])
parser.add_argument('--severity-start', type=int, default=5)
parser.add_argument('--severity-end', type=int, default=5)
parser.add_argument('--output-csv', type=str, default='./outputs/result.csv')
parser.add_argument('--cache-size', type=int, default=128)
parser.add_argument('--use-cache', action='store_true', default=False)
parser.add_argument('--repeat', type=int, default=5)



# === JS-MLA / consistency
parser.add_argument('--mla-tau', type=float, default=1.0)
parser.add_argument('--cons-weight', type=float, default=0.2)
# parser.add_argument('--re-lambda', type=float, default=0.5)
parser.add_argument('--re-lambda', type=float, default=0.)
parser.add_argument('--bal-weight', type=float, default=1.0)
parser.add_argument('--ce-weight', type=float, default=1.0)
parser.add_argument('--ent-weight', type=float, default=1.0)

# === EMA
parser.add_argument('--ema-m', type=float, default=0.999)

# === RAF: Reliability-Aware Fusion (no backprop, no signal rectification)
parser.add_argument('--raf-temp', type=float, default=1.0, help='entropy->reliability temperature')
parser.add_argument('--raf-ema', type=float, default=0.95, help='EMA momentum for entropy baselines')
parser.add_argument('--raf-alpha', type=float, default=0.5, help='baseline correction strength')
parser.add_argument('--raf-use-f', type=float, default=0.3, help='blend factor with multimodal logits_f (hedge)')
parser.add_argument('--raf-js-tau', type=float, default=-1.0, help='if >0, add JS consistency penalty with logits_f')

args = parser.parse_args()
if args.raf_js_tau is not None and args.raf_js_tau <= 0:
    args.raf_js_tau = None
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args)

# =========================
# Dataset classes
# =========================
if args.dataset == 'vggsound':
    args.n_class = 309
elif args.dataset == 'ks50':
    args.n_class = 50

if args.corruption_modality == 'video':
    corruption_list = [
        'defocus_blur','gaussian_noise','shot_noise','impulse_noise','glass_blur',
        'motion_blur','zoom_blur','snow','frost','fog','contrast','brightness',
        'elastic_transform','pixelate','jpeg_compression',
    ]
elif args.corruption_modality == 'audio':
    corruption_list = ['traffic','crowd','rain','wind','thunder','gaussian_noise']
elif args.corruption_modality == 'none':
    corruption_list = ['clean']
    args.severity_start = args.severity_end = 0

# =========================
# Main eval loop
# =========================
mean_acc_list, std_acc_list = [], []

for corruption in corruption_list:
    for severity in range(args.severity_start, args.severity_end+1):
        epoch_accs = []

        if args.corruption_modality == 'none':
            data_val = os.path.join(args.json_root, corruption, f'severity_{severity}.json')
        else:
            data_val = os.path.join(args.json_root, args.corruption_modality, corruption, f'severity_{severity}.json')
        print('===> Now handling: ', data_val)

        for itr in range(1, args.repeat+1):
            seed = int(str(itr)*3)
            seed_everything(seed=seed)
            print(f"### Seed= {seed}, Round {itr} ###")

            im_res = 224
            val_audio_conf = {
                'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                'dataset': args.dataset, 'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std,
                'noise': False, 'im_res': im_res
            }

            tta_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False
            )

            # ==== model ====
            if args.model == 'cav-mae-ft':
                va_model = models.CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
            else:
                raise ValueError('model not supported')

            if args.pretrain_path != 'None':
                mdl_weight = torch.load(args.pretrain_path)
                if not isinstance(va_model, torch.nn.DataParallel):
                    va_model = torch.nn.DataParallel(va_model)
                va_model.load_state_dict(mdl_weight, strict=False)
                print('Loaded pretrained weights from ', args.pretrain_path)
            else:
                warnings.warn("No pre-trained model specified.")

            if not isinstance(va_model, torch.nn.DataParallel):
                va_model = torch.nn.DataParallel(va_model)
            va_model.cuda()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ==== TTA ====
            adapt_flag = args.tta_method != 'None'

            va_model = ABPEM.configure_model(va_model)

            params, _ = ABPEM.collect_params(va_model)
            optimizer = torch.optim.Adam([{'params': params, 'lr': args.lr}], weight_decay=args.wd)

            read_model = ABPEM.READ(va_model, optimizer, device, args)

            if args.use_cache:
                args.cache = ABPEM.Cache(args.cache_size // args.batch_size)

            read_model.eval()
            batch_accs = []
            data_bar = tqdm(tta_loader)
            for i, (a_input, v_input, labels) in enumerate(data_bar):
                a_input, v_input = a_input.to(device), v_input.to(device)
                outputs, loss = read_model((a_input, v_input), adapt_flag=adapt_flag, labels=labels)
                acc = accuracy(outputs[1], labels, topk=(1,))
                acc = round(acc[0].item(), 2)
                batch_accs.append(acc)
                data_bar.set_description(
                    f'Batch#{i}: L0#{loss[0]:.4f}, L1#{loss[1]:.4f}, L2#{loss[2]:.4f}, L3#{loss[3]:.4f}, L4#{loss[4]:.4f}, ACC#{acc:.2f}'
                )

            epoch_acc = round(sum(batch_accs) / len(batch_accs), 2)
            epoch_accs.append(epoch_acc)
            print(f'Epoch: acc={epoch_acc}')

        mean_value = np.round(np.max(epoch_accs), 2)
        std_value = np.round(np.std(epoch_accs), 2)
        mean_acc_list.append(mean_value)
        std_acc_list.append(std_value)
        print(f'===>{corruption}-{severity}, mean: {mean_value}, std: {std_value}')
        with open(args.output_csv, 'a') as f:
            f.write(f'{corruption}, {severity}, {mean_value}, {std_value}\n')

        # # === Evaluate JS-based modality corruption detection accuracy ===
        # if args.corruption_modality in ['audio', 'video', 'none']:
        #     print(f"\n===> Evaluating JS-based modality corruption detection ({args.corruption_modality}) ...")
        #
        #
        #     evaluate_modality_detection(read_model.model, tta_loader, device, args)

print(f'===>AVG, mean: {np.mean(mean_acc_list)}, std: {np.mean(std_acc_list)}')
with open(args.output_csv, 'a') as f:
    f.write(f'AVG, -, {np.mean(mean_acc_list)}, {np.mean(std_acc_list)}\n')
