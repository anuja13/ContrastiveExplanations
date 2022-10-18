import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import copy
import matplotlib.pyplot as plt
import imageio
from matplotlib import pylab as P
import numpy as np
from PIL import Image,ImageFont,ImageDraw


def calculate_outputs_and_gradients(inputs, model, target_label_idx, cuda=False):
    gradients = []
    scores = []
    images = []
    for input in inputs:     
        input = pre_processing(input, cuda)
        output = model(input)
        score = F.softmax(output, dim=1)
        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 0).item()  # 0 corresponds to abnormal class, 1 is Normal

        index = np.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)

        if cuda:
            index = index.cuda()()
        output = output.gather(1, index) # along columns dim 1 gather in
        # clear grad
        model.zero_grad()
        output.backward()
        gradient = input.grad.detach().cpu().numpy()[0]
        gradient[gradient<gradient.mean()] = 0
        gradients.append(gradient)
        scores.append(score[0][target_label_idx].item())
    gradients = np.array(gradients)
    scores = np.array(scores)
    inds_TGT = np.argwhere(scores>=0.5)
    inds_CF = np.argwhere(scores<0.5)
    sorted_grads_target = np.array(gradients[inds_TGT]).squeeze()
    sorted_grads_CF = np.array(gradients[inds_CF]).squeeze()
    # show_grid(inputs, scores)
    return gradients, sorted_grads_target, sorted_grads_CF, scores, inds_TGT,inds_CF

def pre_processing(obs, cuda, model_type='custom'):

    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    obs = obs / 255
    obs = (obs - mean) / std
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    if cuda:
        torch_device = torch.device('cuda:0')
    else:
        torch_device = torch.device('cpu')
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device, requires_grad=True)
    return obs_tensor

    
    
    
def load_unparallel(state_dict):
    # check if the keys are already compatible with data parallel, i.e, have prefix 'module'
    unparallel_dict = copy.deepcopy(state_dict)
    for key in state_dict.keys():
        if 'model' in key or 'classifier' in key:
            new_key = key[7:]
            # print(new_key)
            unparallel_dict[new_key] = unparallel_dict.pop(key)
        else:
            print('already un-parallel')
            break           
    return unparallel_dict


def pred_fun(model, target_label_idx):
    
    def predict(inp):
        _, _, _, scores, _,_ = calculate_outputs_and_gradients(inp, model, target_label_idx, cuda=False)
        # print(scores)
        return scores

    return predict

def VisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.

  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """
  image_2d = np.sum(np.abs(image_3d), axis=2)

  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)

  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def ShowImage(im, title1='', title2=None, ax=None, cmap=None):
    if ax is None:
        P.figure()
    P.axis('off')
    if not cmap:
        P.imshow(im)
    else:
        P.imshow(im, cmap=cmap)
    P.title(title1,loc='left',fontsize=10)
    if title2 != None:
        P.title(title2,loc='right')

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    # P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.imshow(im, cmap='gray')
    P.title(title)

def show_grid(inputs, scores):
    ROWS = int(len(inputs)/5) if len(inputs)>10 else 3
    COLS = 7
    P.figure(figsize=(ROWS , COLS))
    for i,inp in enumerate(inputs):
       # print(scores[i])
       im = inp
       # Render images and scores
       
       ShowImage(im/255, title1="{:.2f}".format(scores[i]), ax=P.subplot(ROWS, COLS, i+1), cmap=None)
    plt.show()

def gif(files,name):
    images = []
    for file in files:
        images.append(file)
    imageio.mimsave(f'./results/{str(name)}.gif', images, duration=0.5)
    
def cosine_sim(grads):
   
    grads=torch.tensor(grads)
    cos = torch.nn.CosineSimilarity(dim=0)
    inds =  [int(grads.shape[0]//2), 1, int(grads.shape[0]-2)]
    for ind in inds:
        print(ind)
        sim=[]
        out_img_grad = torch.flatten(grads[ind])
        for i in range(grads.shape[0]):
            sim.append(cos(out_img_grad,torch.flatten(grads[i])))
            # print(sim)
        plt.plot(range(grads.shape[0]), sim)
    plt.title('cosine similarity')
    plt.savefig(f'./results/cosin/{ind}.png', bbox_inches='tight')
    plt.show()
   
    