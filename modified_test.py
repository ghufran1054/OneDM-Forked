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


import cv2
import torch.nn.functional as F
import numpy as np

laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

def weighted_levenshtein(word1, word2):
    len1, len2 = len(word1), len(word2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        for j in range(len2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:

                dp[i][j] = min(
                    dp[i - 1][j] + 1,          # Deletion
                    dp[i][j - 1] + 1,          # Insertion
                    dp[i - 1][j - 1] + 1    # Substitution
                )
    return dp[len1][len2]

def find_closest_word(word1, target_words):
    closest_word = None
    min_distance = float('inf')

    for word in target_words:
        distance = weighted_levenshtein(word1, word)
        for char in word1:
            if char in word:
                distance -= 0.25
        distance += 0.25 * np.abs(len(word1) - len(word))
        if distance < min_distance:
            min_distance = distance
            closest_word = word

    return closest_word
def laplace_ostu(file):
    image = cv2.imread(file, 1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    y = F.conv2d(x, laplace.repeat(1, 3, 1, 1), stride=1, padding=1, )
    y = y.squeeze().numpy()
    y = np.clip(y, 0, 255)
    y = y.astype(np.uint8)
    ret, threshold = cv2.threshold(y, 0, 255, cv2.THRESH_OTSU)
    return threshold
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
    page_height = len(lines) * line_height + (len(lines) - 1) * vertical_spacing + 1080

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

def post_process_image(image, method="threshold", threshold=100, laplacian_kernel_size=3):
    """
    Sharpen the image using different techniques.

    Parameters:
    - image: PIL Image object.
    - method: The sharpening method to apply ('threshold', 'edge', 'laplacian').
    - threshold: Threshold value for the 'threshold' method.
    - laplacian_kernel_size: Kernel size for the 'laplacian' method.

    Returns:
    - Sharpened PIL Image.
    """
    
    # Convert the PIL image to a NumPy array (grayscale)
    image_np = np.array(image.convert('L'))  # Convert to grayscale ('L' mode)

    if method == "threshold":
        # Apply basic thresholding: if pixel value < threshold, set to 0 (black); else, set to 255 (white)
        _, sharpened = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
        
    elif method == "edge":
        # Apply edge sharpening using a kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])  # Simple edge detection kernel
        sharpened = cv2.filter2D(image_np, -1, kernel)
        
    elif method == "laplacian":
        # Apply Laplacian filtering for sharpening
        sharpened = cv2.Laplacian(image_np, cv2.CV_64F, ksize=laplacian_kernel_size)
        sharpened = cv2.convertScaleAbs(sharpened)  # Convert back to uint8

    else:
        raise ValueError("Invalid method! Choose 'threshold', 'edge', or 'laplacian'.")

    # Convert the NumPy array back to a PIL image
    sharpened_pil = Image.fromarray(sharpened)

    return sharpened_pil
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

    wid = "168"
    style_data_path = f"data/IAM64-new/test/{wid}"
    laplace_data_path = f"data/IAM64_laplace/test/{wid}"
    style_word_list = ['assuredness', 'success', 'with', 'a', 'mid-way', 'charts', 'is', 'couple', '1956']

    # Open style_data_path folder and get the list of images (names) in it
    style_data = os.listdir(style_data_path)

    # Open laplace_data_path folder and get the list of images (names) in it
    laplace_data = os.listdir(laplace_data_path)

    # See which files are present in style_data but not in laplace_data
    missing_files = [f for f in style_data if f not in laplace_data]

    # For each of the missing file we call the function laplace_ostu
    print("Generating laplace images for missing files: ", missing_files)
    for file in missing_files:
        laplace = laplace_ostu(os.path.join(style_data_path, file))
        cv2.imwrite(os.path.join(laplace_data_path, file), laplace)

    

    print('this process handles characters: ', len(temp_texts))

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
    images = []
    for x_text in tqdm(temp_texts, position=0, desc='batch_number'):

        index = -1
        closest = find_closest_word(x_text, style_word_list)
        print("closest word: ", closest)
        index = style_word_list.index(closest)
        style_image = cv2.imread(os.path.join(style_data_path, style_data[index]), flags=0)
        laplace_image = cv2.imread(os.path.join(laplace_data_path, laplace_data[index]), flags=0)

        style_image = style_image/255.0
        laplace_image = laplace_image/255.0

        style_ref = torch.from_numpy(style_image).unsqueeze(0).to(torch.float32)
        laplace_ref = torch.from_numpy(laplace_image).unsqueeze(0).to(torch.float32)

        style_ref_input_fmt = torch.ones([1, style_ref.shape[0], style_ref.shape[1], style_ref.shape[2]], dtype=torch.float32)
        laplace_ref_input_fmt = torch.zeros([1, laplace_ref.shape[0], laplace_ref.shape[1], laplace_ref.shape[2]], dtype=torch.float32)

        style_ref_input_fmt[0, :, :, 0:style_ref.shape[2]] = style_ref
        laplace_ref_input_fmt[0, :, :, 0:laplace_ref.shape[2]] = laplace_ref

        data_val, laplace = style_ref_input_fmt, laplace_ref_input_fmt
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
            out_path = os.path.join(target_dir, wid)
            os.makedirs(out_path, exist_ok=True)
            # image.save(os.path.join(out_path, x_text + ".png"))

            """Optional Part Further reducing the size of the generated text image while preserving aspect ratio"""
            ori_width, ori_height = image.size
            aspect_ratio = ori_width/ori_height
            target_height = 32
            new_width = int(target_height * aspect_ratio)

            # Now resize this PIL image to target_height x new_width
            image = image.resize((new_width, target_height))

            post_process_image(image)


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
