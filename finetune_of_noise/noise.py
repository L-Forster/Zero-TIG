import torch
import torch.nn as nn
import numpy as np
import scipy.io


# for sampling, use actual value of: (vmax-vmin)*label + vmin, as label is scaled to be between 0 and 1
# NOTE: for brightness it is already in the right range, as it isn't a value to be predicted
# and for banding_noise_angle it is either 0 or 1 so no need for scaling
actual_labels_starlight = {
    'alpha_brightness': [0.1, 0.5],  # alpha for brightness
    'gamma_brightness': [0.1, 1],  # gamma for brightness
    'shot_noise': [0, 0.5],
    'read_noise': [0, 0.1],
    'quant_noise': [0, 0.1],
    'band_noise': [0, 0.005],
    'band_noise_temp': [0, 0.005],
    'periodic0': [0, 0.5],
    'periodic1': [0, 0.5],
    'periodic2': [0, 0.5],
}

actual_labels_eld = {
    'alpha_brightness': [0.1, 0.5],  # alpha for brightness
    'gamma_brightness': [0.1, 1],  # gamma for brightness
    'shot_noise_log': [np.log(1e-1), np.log(30)],
    'read_noise_scale': [0, 30],
    'read_noise_tlambda': [np.log(1e-1), np.log(30)], # lmbda=0.14 for a gaussian (refer to paper)
    'quant_noise': [0, 0.1],
    'band_noise': [0, 0.005]
}

def heteroscedastic_noise(x, var_r, var_s, device='cuda'):
    var = x*var_s + var_r
    n_h = torch.randn(x.shape, device=device)*var
    return n_h

def shot_noise(x, k, device='cuda'):
    if x.max() <= 1.0:
        x = (x * 255).int()
        noisy = torch.poisson(x / k) * k
        noisy = noisy.float() / 255.0
    else:
        noisy = torch.poisson(x / k) * k
    return noisy.to(device)

def gaussian_noise(x, scale, loc=0, device='cuda'):
    return torch.randn_like(x) * scale + loc

# REFERENCE: https://github.com/Srameo/LED/blob/main/led/data/noise_utils/common.py
def tukey_lambda_noise(x, scale, t_lambda=1.4, device='cuda'):
    def tukey_lambda_ppf(p, t_lambda):
        assert not torch.any(t_lambda == 0.0)
        return 1 / t_lambda * (p ** t_lambda - (1 - p) ** t_lambda)
    
    tmp_scale = False
    if x.max() <= 1.0:
        x = x * 255
        tmp_scale = True

    epsilon = 1e-10
    U = torch.rand_like(x) * (1 - 2 * epsilon) + epsilon
    Y = tukey_lambda_ppf(U, t_lambda) * scale

    if tmp_scale:
        Y = (Y / 255.0).float()

    return Y

def quant_noise(x, q, device='cuda'):
    return (torch.rand_like(x) - 0.5) * q

def quantization_noise(x, vmax, device='cuda'):
    n_quant = vmax * torch.rand(x.shape, device=device)
    return n_quant

def banding_noise(x, band_params, band_angles, num_frames, device='cuda'):
    x = x.view(-1, num_frames, x.shape[1], x.shape[2], x.shape[3])
    B, N, C, H, W = x.shape
    band_all = []
    
    for i in range(x.shape[0]):
        band_angle = torch.round(band_angles[i][0]).item()
        if band_angle == 0: # horizontal banding
            band_temp = band_params[i] * torch.randn((N, C, H), device=device).unsqueeze(-1) # (NxCxHx1)
            band_temp = band_temp.repeat(1, 1, 1, W).view(N, C, H, W)
            band_all.append(band_temp)
        elif band_angle == 1: # vertical banding
            band_temp = band_params[i] * torch.randn((N, C, W), device=device).unsqueeze(-2) # (NxCx1xW)
            band_temp = band_temp.repeat(1, 1, H, 1).view(N, C, H, W)
            band_all.append(band_temp)
        else:
            raise ValueError("band_angle should be 0 or 1 but got:", band_angle)
    n_band = torch.stack(band_all, dim=0)
    n_band = n_band.view(B*N, C, H, W)
    x = x.view(B*N, C, H, W)

    return n_band

def banding_temp_noise(x, bandt_params, bandt_angles, num_frames, device='cuda'):
    # Banding temp noise
    # NOTE: must do this for individual videos to ensure different angles per video in batch
    x = x.view(-1, num_frames, x.shape[1], x.shape[2], x.shape[3])
    B, N, C, H, W = x.shape
    bandt_all = []
    for i in range(x.shape[0]):
        bandt_angle = torch.round(bandt_angles[i][0]).item()
        if bandt_angle == 0: # horizontal banding
            bandt_temp = bandt_params[i] * torch.randn((C, H), device=device).unsqueeze(-1).unsqueeze(0) # 1xCxHx1
            bandt_temp = bandt_temp.repeat(N, 1, 1, W).view(N, C, H, W)
            bandt_all.append(bandt_temp)
        elif bandt_angle == 1: # vertical banding
            bandt_temp = bandt_params[i] * torch.randn((C, W), device=device).unsqueeze(-2).unsqueeze(0)  # 1xCx1xW
            bandt_temp = bandt_temp.repeat(N, 1, H, 1).view(N, C, H, W)
            bandt_all.append(bandt_temp)
        else:
            raise ValueError("bandt_angles should be 0 or 1 but got:", bandt_angles)
    n_bandt = torch.stack(bandt_all, dim=0)
    n_bandt = n_bandt.view(B*N, C, H, W)
    return n_bandt

def periodic_noise(x, band_angles, param0, param1, param2, num_frames, device='cuda'):
    x = x.view(-1, num_frames, x.shape[1], x.shape[2], x.shape[3])
    B, N, C, H, W = x.shape
    # Create periodic noise separately for each frame based on band_noise_angle
    n_periodic = torch.zeros_like(x, device=device)
    for i in range(x.shape[0]):
        c_periodic = torch.zeros(*x[i].shape,  dtype=torch.cfloat, device=device)
        band_angle = torch.round(band_angles[i][0]).item()
        if band_angle == 0:
            c_periodic[...,0,0] = param0[i][0]*torch.randn((x.shape[1:-2]), device=device)
            periodic0 = param1[i][0]*torch.randn((x.shape[1:-2]), device=device)
            periodic1 = param2[i][0]*torch.randn((x.shape[1:-2]), device=device) 
            c_periodic[...,x.shape[-2]//4,0] = torch.complex(periodic0, periodic1)
            c_periodic[...,3*x.shape[-2]//4,0] = torch.complex(periodic0, -periodic1)
        elif band_angle == 1:
            c_periodic[...,0,0] = param0[i][0]*torch.randn((x.shape[1:-2]), device=device)
            periodic0 = param1[i][0]*torch.randn((x.shape[1:-2]), device=device)
            periodic1 = param2[i][0]*torch.randn((x.shape[1:-2]), device=device) 
            c_periodic[...,0,x.shape[-1]//4] = torch.complex(periodic0, periodic1)
            c_periodic[...,0,3*x.shape[-1]//4] = torch.complex(periodic0, -periodic1)
        n_periodic[i] = torch.abs(torch.fft.ifft2(c_periodic, norm="ortho"))
    x = x.view(B*N, C, H, W)
    n_periodic = n_periodic.view(B*N, C, H, W)

    return n_periodic


def reshape_noise_params(noise_params, noise_model, num_frames=1):
    # Reshape noise parameters for batch processing
    if noise_model == "starlight":
        noise_list = 'shot_read_quant_band1_bandt_periodic'
    elif noise_model == "eld":
        noise_list = 'shotLog_readTL_quant_band1'
    noise_dict = {}
    bs = noise_params.shape[0]  # batch size

    # Add brightness params
    noise_dict['alpha_brightness'] = noise_params[:, 0].view(bs, 1, 1, 1)
    noise_dict['gamma_brightness'] = noise_params[:, 1].view(bs, 1, 1, 1)

    # Add noise params
    band_count = 0
    total_band_count = noise_list.count('band') 
    # for i, noise in enumerate(noise_list.split('_'), start=2):  # start=2 to skip brightness params
    i = 2
    for noise in noise_list.split('_'):
        key = noise+'_noise'
        if noise == 'periodic':
            noise_dict['periodic0'] = noise_params[:,i].view(bs, 1)
            noise_dict['periodic1'] = noise_params[:,i+1].view(bs, 1)
            noise_dict['periodic2'] = noise_params[:,i+2].view(bs, 1)
            i += 2
        elif 'band' in noise:
            if noise == 'band1':
                noise_dict['band_noise'] = noise_params[:,i].view(bs, 1, 1, 1)
                band_count += 1
            elif noise == 'bandt':
                noise_dict['band_noise_temp'] = noise_params[:,i].view(bs, 1, 1, 1)
                band_count += 1
            # Add band angle only once
            if band_count == total_band_count:
                noise_dict['band_noise_angle'] = noise_params[:,i+1].view(bs, 1)
                i += 1
        elif noise == 'readTL':
            noise_dict['read_noise_scale'] = noise_params[:,i].view(bs, 1, 1, 1)
            noise_dict['read_noise_tlambda'] = noise_params[:,i+1].view(bs, 1, 1, 1)
            i += 1
        elif noise == 'shotLog':
            noise_dict['shot_noise_log'] = noise_params[:,i].view(bs, 1, 1, 1)
        else:
            noise_dict[key] = noise_params[:,i].view(bs, 1, 1, 1)

        i += 1

    return noise_dict


def StarlightNoise(x, noise_dict, num_frames=1, device='cuda'):
    noise = torch.zeros_like(x, device=device)
    noise += heteroscedastic_noise(x, noise_dict['shot_noise'], noise_dict['read_noise'], device=device)
    noise += quantization_noise(x, noise_dict['quant_noise'], device=device)
    noise += banding_noise(x, noise_dict['band_noise'], noise_dict['band_noise_angle'], num_frames, device=device)
    noise += banding_temp_noise(x, noise_dict['band_noise_temp'], noise_dict['band_noise_angle'], num_frames, device=device)
    noise += periodic_noise(x, noise_dict['band_noise_angle'], noise_dict['periodic0'], noise_dict['periodic1'], noise_dict['periodic2'], num_frames, device=device)
    
    return x + noise

def ELDNoise(x, noise_dict, num_frames=1, device='cuda'):
    noisy = shot_noise(x, noise_dict['shot_noise'], device=device)
    noisy += tukey_lambda_noise(x, noise_dict['read_noise_scale'], noise_dict['read_noise_tlambda'], device=device)
    noisy += quant_noise(x, noise_dict['quant_noise'], device=device)
    noisy += banding_noise(x, noise_dict['band_noise'], noise_dict['band_noise_angle'], num_frames, device=device)
    return noisy


def generate_noise(x, noise_dict_not_scaled, noise_model, num_frames=1, device='cuda'):
    assert x.min() >= 0 and x.max() <= 1, "Input tensor should be in [0, 1] range"
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    assert x.ndim == 4 and x.shape[0] % num_frames == 0, f"Input tensor should have shape [B*N, C, H, W] but got: {x.shape} for N={num_frames}"

    # Scale noise dict back to actual values: (vmax-vmin)*label + vmin
    noise_dict = {}
    actual_labels = actual_labels_starlight if noise_model == "starlight" else actual_labels_eld
    for key in actual_labels:
        if key == 'shot_noise_log': # scale log_K to be between log(1e-1) and log(30)
            scale = actual_labels[key][1] - actual_labels[key][0]
            log_K = scale*noise_dict_not_scaled[key] + actual_labels[key][0]
            noise_dict['shot_noise'] = torch.exp(log_K)
        elif key == 'read_noise_tlambda': # scale log_K to be between log(1e-1) and log(30)
            scale = actual_labels[key][1] - actual_labels[key][0]
            log_lmbda = scale*noise_dict_not_scaled[key] + actual_labels[key][0]
            noise_dict['read_noise_tlambda'] = torch.exp(log_lmbda)
        else:
            scale = actual_labels[key][1] - actual_labels[key][0]
            noise_dict[key] = scale*noise_dict_not_scaled[key] + actual_labels[key][0]
    # Add band_noise_angle
    noise_dict['band_noise_angle'] = torch.round(noise_dict_not_scaled['band_noise_angle']).view(-1, 1)

    # Put labels onto device
    for key in noise_dict:
        noise_dict[key] = noise_dict[key].to(device)

    # Adjust brightness
    alpha = noise_dict['alpha_brightness'] # NOTE: ELD also scales images separately to K!
    gamma = noise_dict['gamma_brightness']
    x = alpha*(torch.pow(x, 1/gamma))

    if noise_model == "starlight":
        noisy = StarlightNoise(x, noise_dict, num_frames=num_frames, device=device)
    elif noise_model == "eld":
        noisy = ELDNoise(x, noise_dict, num_frames=num_frames, device=device)
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")

    noisy = torch.clip(noisy, 0, 1)
    
    if squeeze:
        x = x.squeeze(0)
        noisy = noisy.squeeze(0)

    return noisy


if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Load clean images
    clean_1 = Image.open("clean_0.png").convert('RGB')
    clean_2 = Image.open("clean_1.png").convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    clean_1 = transform(clean_1).unsqueeze(0)
    clean_2 = transform(clean_2).unsqueeze(0)
    clean_1 = clean_1.repeat(16, 1, 1, 1) # (N, C, H, W)
    clean_2 = clean_2.repeat(16, 1, 1, 1) # (N, C, H, W)
    # clean = torch.cat([clean_1, clean_2], dim=0)
    # clean = clean.view(2*16, 3, 256, 256) # (B*N, C, H, W)

    # # Load noise parameters
    # # preds_1 = [[0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 0, 0.5, 0.5, 0.5] for _ in range(16)]
    # # preds_2 = [[0.3, 1.0, 0., 0., 0., 1, 1, 1, 0., 0., 0.] for _ in range(16)]
    # preds_1 = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 1, 0] for _ in range(16)]
    # preds_2 = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 1, 0] for _ in range(16)]
    # noise_params = torch.tensor([preds_1, preds_2]).to('cuda')
    # noise_params = noise_params.view(2*16, len(preds_1[0])) # (B*N, L)
    # bs = 2*16

    # # Reshape noise parameters
    # noise_model = "eld"  # or "eld"
    # noise_dict = reshape_noise_params(noise_params, noise_model, num_frames=16)

    # # Generate noise
    # noisy = generate_noise(clean, noise_dict, noise_model, num_frames=16, device='cuda')
    # noisy = noisy.cpu().detach().numpy()
    # clean = clean.cpu().detach().numpy()

    # # Display images
    # fig, axs = plt.subplots(2, 16, figsize=(20, 5))
    # for i in range(16):
    #     axs[0, i].imshow(clean[i].transpose(1, 2, 0))
    #     axs[0, i].axis('off')
    #     axs[1, i].imshow(noisy[i].transpose(1, 2, 0))
    #     axs[1, i].axis('off')
    # plt.tight_layout()
    # plt.show()

    clean = clean_1
    clean = clean.view(16, 3, 256, 256) # (B*N, C, H, W)

    # Load noise parameters
    l = actual_labels_eld
    preds_1 = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0] for _ in range(16)] # NOTE: scaled in generate_noise()
    # preds_1 = [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] for _ in range(16)] # NOTE: scaled in generate_noise()
    noise_params = torch.tensor(preds_1).to('cpu')
    noise_params = noise_params.view(16, len(preds_1[0])) # (B*N, L)
    bs = 16

    # Reshape noise parameters
    noise_model = "eld"  # or "starlight"
    noise_dict = reshape_noise_params(noise_params, noise_model, num_frames=16)

    # Generate noise
    noisy = generate_noise(clean, noise_dict, noise_model, num_frames=16, device='cpu')
    noisy = noisy.detach().numpy()
    clean = clean.detach().numpy()

    # Display images
    plt.imshow(noisy[0].transpose(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    


    
    