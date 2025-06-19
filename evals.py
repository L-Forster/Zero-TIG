import os
import sys
import numpy as np
import torch
import argparse
import logging
import torch.utils
from PIL import Image
from torch.autograd import Variable
from model.model import Finetunemodel, Network
from dataloader.create_data import CreateDataset
# from thop import profile
import cv2
from skimage.metrics import structural_similarity as ssim
import lpips
from torchvision import transforms
from skimage import exposure
from utils.utils import sequential_judgment
import json

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print("Using device:", device)

parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--lowlight_images_path', type=str, default='./lowlight_dataset',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,
                    default='./results/BVI-RLV',
                    help='location of the data corpus')
parser.add_argument('--model_pretrain', type=str,
                    default=r'./weights_1.pt',
                    help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--of_scale', type=int, default=3, help='downscale size when compute OF')
parser.add_argument('--dataset', type=str, default='RLV', help='Specified data set')
parser.add_argument('--gain', type=int, default=100, help='OF loss gain')
parser.add_argument('--name', type=str, default='run', help='A name for the evaluation run, used for log and metric file names.')


args = parser.parse_args()
save_path = args.save
os.makedirs(save_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
mertic = logging.FileHandler(os.path.join(args.save, f'{args.name}.log'))
mertic.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(mertic)

logging.info("train file name = %s", os.path.split(__file__))
TestDataset = CreateDataset(args, task='test')
test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)
print("Total image number: ", str(TestDataset.__len__()))

logging.info("Model path = %s", str(args.model_pretrain))


def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


# load pre-trained weights
lpips_fn = lpips.LPIPS(net='vgg').to(device)


def calc_metrics(lpips_fn, img_array: np.ndarray, gt_array: np.ndarray):
    # LPIPS
    img_tensor = cvt_array2tensor(img_array)
    gt_tensor = cvt_array2tensor(gt_array)
    distance = lpips_fn(img_tensor, gt_tensor).item()

    # PSNR
    img = np.round(img_array * 255).astype(np.uint8)
    gt = np.round(gt_array * 255).astype(np.uint8)
    psnr = cv2.PSNR(img, gt)
    # SSIM
    ssim_index = ssim(img, gt, multichannel=True, channel_axis=2, data_range=255)

    return psnr, ssim_index, distance


def cvt_array2tensor(arr):
    data = torch.from_numpy(arr).float()
    data = data.permute(2, 0, 1)
    data = (data - 0.5) * 2
    data = data.to(device).unsqueeze(0)

    return data

def histogram_match(out: np.ndarray, gt: np.ndarray):
    matched_out = exposure.match_histograms(out, gt) # float
    # matched_out = np.round(matched_out * 255).astype(np.uint8)
    return matched_out

def main():
    # metric
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_psnr_hm = 0
    total_ssim_hm = 0
    total_lpips_hm = 0
    num_img = 0
    is_hist_match = True
    is_save_img = True

    model = Finetunemodel(args)
    model = model.to(device)
    model.eval()
    # Calculate model size
    total_params = calculate_model_parameters(model)
    print("Total number of parameters: ", total_params)
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        for i, (input, img_name, img_path, last_img_path) in enumerate(test_queue):
            model.is_new_seq = sequential_judgment(img_path[0], last_img_path[0])
            if model.is_new_seq:
                print("Eval Get this img from: ", img_path, "\n Last img from: ", last_img_path)

            input = Variable(input).to(device)
            input_name = img_name[0].split('/')[-1].split('.')[0]
            gt_path = img_path[0].replace('input', 'gt').replace('low_light_', 'normal_light_')
            splits = img_path[0].split(os.sep)
            data_source = splits[-3] +'/'+splits[-2]

            gt_img = np.asarray(Image.open(gt_path, mode='r'), dtype=np.float32) / 255.

            # inference start from here
            enhance, output, illum = model(input)
            # last_H3 = save_images(model.last_H3)
            # Image.fromarray(last_H3).save('./' + input_name + '_denoise_last' + '.png', 'PNG')
            model.update_H3(output, illum)

            # cvt data format
            output_array = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr, ssim_index, lpips_value = calc_metrics(lpips_fn, output_array, gt_img)
            total_psnr += psnr
            total_ssim += ssim_index
            total_lpips += lpips_value
            num_img += 1

            print(f"NUM: {num_img}, PSNR: {psnr:.3f}, SSIM: {ssim_index:.3f}, LPIPS: {lpips_value:.3f}")
            print(
                f"Total PSNR: {(total_psnr / num_img):.3f}, Total SSIM: {(total_ssim / num_img):.3f}, Total LPIPS: {(total_lpips / num_img):.3f}")

            if is_hist_match:
                output_array = histogram_match(output_array, gt_img)
                psnr_hm, ssim_index_hm, lpips_value_hm = calc_metrics(lpips_fn, output_array, gt_img)
                total_psnr_hm += psnr_hm
                total_ssim_hm += ssim_index_hm
                total_lpips_hm += lpips_value_hm
                print(f"NUM: {num_img}, PSNR_HM: {psnr_hm:.3f}, SSIM_HM: {ssim_index_hm:.3f}, LPIPS_HM: {lpips_value_hm:.3f}")
                print(
                    f"Total PSNR_HM: {(total_psnr_hm / num_img):.3f}, Total SSIM_HM: {(total_ssim_hm / num_img):.3f}, Total LPIPS_HM: {(total_lpips_hm / num_img):.3f}")


            # save
            if is_save_img and i < 20:
                save_dir = os.path.join(args.save, data_source)
                os.makedirs(save_dir, exist_ok=True)
                input_name = '%s' % (input_name)
                enhance = save_images(enhance)
                output = save_images(output)
                # os.makedirs(args.save + '/result', exist_ok=True)
                Image.fromarray(output).save(save_dir + '/' + input_name + '_denoise' + '.png', 'PNG')
                Image.fromarray(enhance).save(save_dir + '/' + input_name + '_enhance' + '.png', 'PNG')
                if is_hist_match:
                    # histogram matching results
                    save_img = cv2.cvtColor(np.round(output_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_dir, input_name+'_denoise_hm.png'), save_img)

    torch.set_grad_enabled(True)
    with open(os.path.join(args.save, f'{args.name}_Metrics.json'), 'w') as file:
        json.dump({
                   'Total_PSNR': total_psnr/num_img,
                   'Total_SSIM': total_ssim/num_img,
                   'Total_LPIPS': total_lpips/num_img,
                   'Total_PSNR_HM': total_psnr_hm/num_img,
                   'Total_SSIM_HM': total_ssim_hm/num_img,
                   'Total_LPIPS_HM': total_lpips_hm/num_img},
                  file)


if __name__ == '__main__':
    main()
