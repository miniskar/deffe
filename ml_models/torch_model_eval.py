"""
  model_eval.py: a model evalulation routine
  
   but the loss function is hardcoded
"""

import numpy as np
import pdb
import torch
import torch.nn.functional as F

from tqdm import tqdm, trange


def model_eval(args, net, loader, device, n_val, loss_fn):
    """
      Running model inference and compute the validation accuracy
    """
    net.eval()

    tot = 0.0
    accum = 0 
    with tqdm(total=n_val, desc='validation batches', unit='sample', ascii=True, dynamic_ncols=True, leave=False) as pbar:
        for batch in loader:
            features = batch[0]
            obj = batch[1]
            tmp = obj.shape
            bsize= tmp[0]
            n_obj = tmp[-1]
            obj = obj.view(bsize, n_obj)
            
            features = features.to( device=device, dtype=torch.float32 )
            ground_truth = obj.to( device=device, dtype=torch.float32 )

            pred = net(features)
            for ground_truth, pred in zip(ground_truth, pred):
                if args.real_objective:
                    #tot += F.mse_loss( pred.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0) ).item()
                    tot += torch.flatten(torch.abs(torch.exp(pred.unsqueeze(dim=0))-torch.exp(ground_truth.unsqueeze(dim=0)))/torch.exp(ground_truth.unsqueeze(dim=0)), start_dim=0).tolist()[0]
                else:
                    tot += F.mse_loss( pred.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0) ).item()


            accum += features.shape[0]
            pbar.set_postfix( **{'val loss': tot/accum} )
            pbar.update( features.shape[0] )
            
    return tot/n_val


def model_compute_error(args, net, loader, device, n_val):
    """
      Compute the relative error of the model prediction and ground truth
      Returns the ground truth and the predicted results
    """
    net.eval()

    first_batch = True
    all_obj = None
    all_pred = None
    for batch in loader:
        features = batch[0]
        obj = batch[1]
        tmp = obj.shape
        bsize= tmp[0]
        n_obj = tmp[-1]
        obj = obj.view(bsize, n_obj)

        features = features.to( device=device, dtype=torch.float32 )
        pred = net(features)
        pred = pred.to('cpu').detach().numpy()
        obj = obj.numpy()

        if first_batch:
            all_obj = obj
            all_pred = pred
            first_batch = False
        else:
            all_obj = np.append(all_obj, obj, axis=0)
            all_pred = np.append(all_pred, pred, axis=0)

        pass

    if args.real_objective:
        all_obj = np.exp(all_obj)
        all_pred = np.exp(all_pred)
    difference = all_obj - all_pred
    abs_error = abs( 100. * difference/(all_obj+1e-8) ) 

    if False:
        print(all_obj.shape)
        print(all_pred.shape)
        
        print(difference)
        print(abs_error)

        
    output={'max':np.amax(abs_error),
            'min':np.amin(abs_error),
            'avg':np.mean(abs_error),
            'std':np.std(abs_error)}

    #print("Output: "+str(output))
    return output
