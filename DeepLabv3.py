import cv2
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def deeplabv3(input_path, output_path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    input_path = '/data/VFI_Database/01_vimeo_triplet/'
    output_path = '../DeepLabv3_output/'

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    frame_txt = os.path.join(input_path, 'tri_trainlist.txt')
    with open(frame_txt, 'r') as f:
        trainlist = f.read().splitlines()


    limit = len(trainlist)
    for i in range(0, limit, 1):
        frame_path = trainlist[i]
        frame_path_ = os.path.join(input_path, 'sequences', frame_path + '/im1.png')
        img0 = Image.open(frame_path_)
        img0 = img0.convert("RGB")

        output_frame_path = os.path.join(output_path, frame_path)
        if not os.path.exists(output_frame_path):
            os.makedirs(output_frame_path)

        input_tensor = preprocess(img0)
        input_batch = input_tensor.unsqueeze(0) # 모델이 원하는 미니 배치를 만듭니다.

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
           # out = model(img0)
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(img0.size)
        r.putpalette(colors)

        r.save(os.path.join(output_frame_path + '/img0.png'), 'png')

        print('save img')

if __name__ == '__main__':
    deeplabv3(input_path = '/data/VFI_Database/01_vimeo_triplet/', output_path = '../DeepLabv3_output/')