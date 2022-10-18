import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib import pylab as P
from model import Resnet50
from utils import calculate_outputs_and_gradients, ShowImage, VisualizeImageGrayscale, load_unparallel
from xai_util import  get_explanation, get_gradients
from visualization import visualize
import argparse
import os
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16

parser = argparse.ArgumentParser(description='Causal and Non causal explanations for Capsule Endoscopy.')
parser.add_argument('--cuda', action='store_true', help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='custom', help='the type of network')
parser.add_argument('--checkpoint_path', type= str, default='/home/user1/PhD_CAPSULEAI/XAI/ContrastiveExplainations/checkpoints/3_model.pth', help= 'path to checkpoints for pretrained stylegan model')
parser.add_argument('--imgs_path', type=str, default='./examples', help='path to query images')
parser.add_argument('--target_index', type=int, default=0, help='label index of chosen class, 0 for abnormal and 1 normal for this model')


if __name__ == '__main__':
    args = parser.parse_args()
    # check if have the space to save the results
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.model_type):
        os.mkdir('results/' + args.model_type)
    
    # Load model
    model = Resnet50(num_classes=2)
    checkpoint = torch.load(args.checkpoint_path)
    print('Loading trained model weights...')
    try:
        model.load_state_dict(checkpoint)
    except:
        unparallel_dict = load_unparallel(checkpoint)
        model.load_state_dict(unparallel_dict)

    model.eval()
    if args.cuda:
        model.cuda()

    target_label_idx = args.target_index 
    for dir in os.listdir(args.imgs_path):
        print('UC Biomarker ID : ',dir)
        img_names = sorted(os.listdir(os.path.join(args.imgs_path, dir)))
        images= [cv2.imread(os.path.join(args.imgs_path, dir,img_names[i])) for i in range(len(img_names))]
        scaled_images = [cv2.resize(img, (224, 224)).astype(np.float32) for img in images]
        scaled_images = [img[:, :, (2, 1, 0)] for img in scaled_images]
        # read the image to explain. This corresponds to the image in each folder (maximal abnormality).
        attributions, counterfactual, semifactual, scores = get_explanation(scaled_images, model, target_label_idx, calculate_outputs_and_gradients, \
                                                        steps=5, num_random_trials=1, cuda=args.cuda,dir=dir)
       
        img = scaled_images[-1]


        saliency_grad = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
        saliency_grad = VisualizeImageGrayscale(saliency_grad)
        saliency_map_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, \
                                                overlay=True, mask_mode=True)

        ROWS = 1
        COLS = 4
        UPSCALE_FACTOR = 30
        P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
        
        # Render the saliency masks.
        ShowImage(img/255, title1='Query Image', ax=P.subplot(ROWS, COLS, 3), cmap=None)
        ShowImage(saliency_map_overlay/255*50, title1='Saliency map', ax=P.subplot(ROWS, COLS, 4), cmap='gray')
        ShowImage(counterfactual/255, title1='Counterfactal', ax=P.subplot(ROWS, COLS, 1), cmap=None)
        ShowImage(semifactual[0]/255, title1='Semi Factual', ax=P.subplot(ROWS, COLS, 2), cmap=None)
        plt.show()
