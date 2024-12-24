import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import Random_StyleIAMDataset, ContentData, generate_type
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from utils.util import fix_seed

device = torch.cuda() if torch.cuda.is_available() else torch.device('cpu')

from PIL import Image

def arrange_images_in_page(images, page_width, line_height, horizontal_spacing=30, vertical_spacing=20):
    """
    Arrange images into a fixed-width "page" with line wrapping, including spacing between words and lines.

    Args:
        images: List of PIL Image objects.
        page_width: Fixed width of the page.
        line_height: Height of each line (height of images + vertical spacing).
        horizontal_spacing: Space between images in the same line.
        vertical_spacing: Space between lines.

    Returns:
        A single PIL Image object with arranged images and spacing.
    """
    # Initialize variables for line tracking
    lines = []
    current_line = []
    current_width = 0

    # Arrange images into lines
    for img in images:
        img_width, img_height = img.size
        # Check if the image fits in the current line (accounting for spacing)
        if current_width + img_width + (len(current_line) * horizontal_spacing) > page_width:
            # Start a new line
            lines.append(current_line)
            current_line = []
            current_width = 0
        # Add the image to the current line
        current_line.append(img)
        current_width += img_width

    # Add the last line if it contains any images
    if current_line:
        lines.append(current_line)

    # Calculate total page height
    page_height = len(lines) * line_height + (len(lines) - 1) * vertical_spacing + 40

    # Create a new blank image for the page
    page_image = Image.new("RGB", (page_width + 30, page_height), (255, 255, 255))

    # Paste images into the page
    y_offset = 20
    for line in lines:
        x_offset = 30
        for i, img in enumerate(line):
            # Paste image
            page_image.paste(img, (x_offset, y_offset))
            x_offset += img.size[0] + horizontal_spacing  # Add spacing after each image
        y_offset += line_height + vertical_spacing  # Move down to the next line

    return page_image
def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    load_content = ContentData()

    text_corpus = generate_type[opt.generate_type][1]
    with open(text_corpus, 'r') as _f:
        texts = _f.read().split()

    temp_texts = texts  # No need to split the data for single CPU inference

    """setup data_loader instances"""
    style_dataset = Random_StyleIAMDataset(os.path.join(cfg.DATA_LOADER.STYLE_PATH, generate_type[opt.generate_type][0]), 
                                           os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0]), len(temp_texts))

    print('this process handles characters: ', len(style_dataset))
    style_loader = torch.utils.data.DataLoader(style_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                                pin_memory=True)

    target_dir = os.path.join(opt.save_dir, opt.generate_type)

    diffusion = Diffusion(device=device)  # Ensure Diffusion runs on CPU

    """build model architecture"""
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS, 
        model_channels=cfg.MODEL.EMB_DIM, 
        out_channels=cfg.MODEL.OUT_CHANNELS, 
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
        attention_resolutions=(1, 1), 
        channel_mult=(1, 1), 
        num_heads=cfg.MODEL.NUM_HEADS, 
        context_dim=cfg.MODEL.EMB_DIM
    ).to(device)

    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0: 
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location=device))
        print('Loaded pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('Input the correct checkpoint path')
    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)

    """generate the handwriting datasets"""
    loader_iter = iter(style_loader)
    images = []
    for x_text in tqdm(temp_texts, position=0, desc='batch_number'):
        data = next(loader_iter)
        data_val, laplace, wid = data['style'][0], data['laplace'][0], data['wid']
        
        # Print Info ABout this dat_val, laplace, wid
        # ONe thing found out these are lists
        # print(len(data_val), len(laplace), len(wid))
        # print((data_val.shape), (laplace.shape), (wid[0]))

        data_loader = []
        if len(data_val) > 224:
            data_loader.append((data_val[:224], laplace[:224], wid[:224]))
            data_loader.append((data_val[224:], laplace[224:], wid[224:]))
        else:
            data_loader.append((data_val, laplace, wid))
        
        for (data_val, laplace, wid) in data_loader:
            style_input = data_val.to(device)
            laplace = laplace.to(device)
            text_ref = load_content.get_content(x_text)
            text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_input.shape[2]//8, (text_ref.shape[1]*32)//8)).to(device)

            if opt.sample_method == 'ddim':
                ema_sampled_images = diffusion.ddim_sample(unet, vae, style_input.shape[0], 
                                                           x, style_input, laplace, text_ref,
                                                           opt.sampling_timesteps, opt.eta)
            elif opt.sample_method == 'ddpm':
                ema_sampled_images = diffusion.ddpm_sample(unet, vae, style_input.shape[0], 
                                                           x, style_input, laplace, text_ref)
            else:
                raise ValueError('Sample method is not supported')
            
            for index in range(len(ema_sampled_images)):
                im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                image = im.convert("L")
                out_path = os.path.join(target_dir, wid[index][0])
                os.makedirs(out_path, exist_ok=True)
                image.save(os.path.join(out_path, x_text + ".png"))
                images.append(image)
    
    # Stich all the images width wise to get a single image
    page_image = arrange_images_in_page(images, 720, images[0].height)
    page_image.save(os.path.join(target_dir, 'page.png'))

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated', help='Target directory for storing generated characters')
    parser.add_argument('--one_dm', dest='one_dm', default='', required=True, help='Pre-trained model for generating')
    parser.add_argument('--generate_type', dest='generate_type', required=True, help='Four generation settings: iv_s, iv_u, oov_s, oov_u')
    parser.add_argument('--device', type=str, default='cpu', help='Device for testing')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=10)
    parser.add_argument('--sample_method', type=str, default='ddim', help='Choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    opt = parser.parse_args()
    main(opt)
