import numpy as np
import torch
from utils import pre_processing,gif, cosine_sim, VisualizeImageGrayscale
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


def get_gradients(scaled_inputs, model, target_label_idx, predict_and_gradients, steps=50, cuda=False,dir=dir):

    all_grads, grads_tgt, grads_CF, scores, inds_TGT,inds_CF = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
    diff_maps = find_difference(scaled_inputs)

    for i in range(all_grads.shape[0]):
        all_grads[-i] = all_grads[-i] * diff_maps[-i]

    # print(scores, inds_TGT, inds_CF)
   
    grads = (all_grads[-1] + all_grads[1:])/2 
    n = 2 # neighbors around
    avg_grads = np.average(grads, axis=0) 
    
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    
    baseline = scaled_inputs[inds_TGT[0][0]]
    input = scaled_inputs[-1]
    delta_X = (pre_processing(input, cuda) - pre_processing(baseline, cuda)).detach().squeeze(0).cpu().numpy()
    delta_X = np.transpose(delta_X, (1, 2, 0)) # TODO MULTIPLY GRADS BY SCORE
    integrated_grad = delta_X * avg_grads 
    
    try:
        counterfactual =  scaled_inputs[inds_CF[-1][0]]
        semifactual = (scaled_inputs[inds_TGT[0][0]],scaled_inputs[inds_TGT[-2][0]])
    except:
        counterfactual = None
        semifactual = None

    return integrated_grad, counterfactual, semifactual, scores



def get_explanation(inputs, model, target_label_idx, predict_and_gradients, steps, num_random_trials, cuda, dir=dir):
    all_intgrads = []
    for i in range(num_random_trials):
        weighted_grad, counterfactuals, semifactuals, scores = get_gradients(inputs, model, target_label_idx,
                                               predict_and_gradients =predict_and_gradients, steps=steps, cuda=cuda, dir=dir)
        all_intgrads.append(weighted_grad)
        print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)

    return avg_intgrads, counterfactuals, semifactuals, scores

def im_show(im, ind, norm2show=0):
    # for scaled input as WCE images:
    if not norm2show:
        im = im/255
        im = np.clip(im, 0, 1)
        plt.imshow(im)
    else:
    # for scaled images with random/black baseline
        plt.imshow(np.clip(im/255,0,1))
    plt.title('scaled input '+str(ind))
    plt.show() 
    
def gaussian_blur(image, sigma):
  """Returns Gaussian blur filtered 3d (WxHxC) image.

  Args:
    image: 3 dimensional ndarray / input image (W x H x C).
    sigma: Standard deviation for Gaussian blur kernel.
  """
  if sigma == 0:
    return image
  return ndimage.gaussian_filter(
      image, sigma=[sigma, sigma, 0], mode="constant")

def plot(all_grads,grads_tgt,grads_CF, scores):
    alphas = np.linspace(0, 1, len(scores))
    fig,ax1 = plt.subplots(figsize=(12,8))
    ax1.plot(alphas, scores, color="orange")
    ax1.set_ylabel("scores",color="orange",fontsize=14)
    ax2=ax1.twinx()
    grads = np.average(all_grads, axis=(1,2,3))
    ax2.plot(alphas, grads, color="blue")
    ax2.set_ylabel("gradients",color="blue",fontsize=14)
    ax2.set_ylim(min(grads), 2*max(grads))
    ax1.set_xlabel("alphas", fontsize=14)
    plt.title(f"scores vs grads", fontsize=14)
    plt.show()




def find_difference(images):
    query = images[-1]
    diff_maps = []
    for i in range(len(images)-1):
        diff_3d= (query-images[i])
        
        noise_2d = np.sum(np.abs(diff_3d), axis=2)
    
        vmax = np.max(noise_2d)
        vmin = np.min(noise_2d)
    
        noise_2d=np.clip((noise_2d - vmin) / (vmax - vmin), 0, 1)
    
        x = np.arange(0, 224, 1)
        y = np.arange(0, 224, 1)
        X, Y = np.meshgrid(x, y)
        Z = noise_2d
    
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.contour3D(X, Y, Z, 50, cmap='Reds')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.axis('off')
        # ax.set_title('3D contour')
        # plt.show()
        diff_map = noise_2d*VisualizeImageGrayscale(query)
        # plt.imshow(diff_map, cmap='gray')
        # plt.show()
        diff_maps.append(diff_map)
    return diff_maps
   





    
    
    
    