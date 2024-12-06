import torch, cv2
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)

def laplace_ostu(file):
    image = cv2.imread(file, 1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize image to 64 height, 128 width
    img = cv2.resize(img, (128, 64))

    # Also save this image as png
    cv2.imwrite('resized_image.png', img)
    x = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).float()
    y = F.conv2d(x, laplace.repeat(1, 3, 1, 1), stride=1, padding=1, )
    y = y.squeeze().numpy()
    y = np.clip(y, 0, 255)
    y = y.astype(np.uint8)
    ret, threshold = cv2.threshold(y, 0, 255, cv2.THRESH_OTSU)
    return threshold

if __name__ == '__main__':
    source_file = 'my_style_1.png'
    saved_img = 'my_style_1_laplace.png'
    threshold = laplace_ostu(source_file)
    cv2.imwrite(saved_img, threshold)