import torch
import torchvision
import numpy as np
import cv2

##########################################################################
## Translation Invariance ################################################
##########################################################################

def extract_min_img(img):
    try:
        img = torch.squeeze(img) * 255
        img_np = img.cpu().numpy().astype(np.uint8)
        retVal, threshold_image = cv2.threshold(img_np,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # this check isn't always reliable, hence the try-except block
        if len(np.array(contours).shape) < 4:
            return None
        contours = np.squeeze(np.array(contours), 0)
        x,y,w,h = cv2.boundingRect(contours)
        bounded_img = img_np[y:y+h, x:x+w]
    except:
        return None
    return bounded_img

def pad_img(target_img, original_img, offsets=[0,0]):
    """
    bounded_img: Image to be padded / inserted into a blank array of zeros
    original_img: Reference img with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(original_img.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + target_img.shape[dim]) for dim in range(target_img.ndim)]
    # Insert the array in the result at the specified offsets
    result[tuple(insertHere)] = target_img
    return result

def generate_all_offsets(target_img, original_img, max_offsets):
    offset_imgs = []
    for dim0 in range(max_offsets[0]):
        for dim1 in range(max_offsets[1]):
            offsets = [dim0, dim1]
            offset_img = pad_img(target_img, original_img, offsets)
            offset_imgs.append(offset_img)
    return np.array(offset_imgs)

def np_to_pytorch_imgs(inputs, channel_first=True):
    if channel_first:
        if len(inputs.shape) == 3: 
            inputs = np.expand_dims(inputs, 1)
        elif len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, 0)
    else:
        inputs = np.expand_dims(inputs, -1)
        
    inputs_torch = torch.from_numpy(inputs).to(torch.float)
    inputs_torch /= 255
    return inputs_torch

def generate_all_translations(inputs, targets):
    
    new_inputs = []
    new_targets = []
    
    for i, (instance, label) in enumerate(zip(inputs, targets)):
        instance_np = np.squeeze(instance.cpu().numpy().astype(np.uint8))
        
        # Locate digit with bounding box
        bounded_img_np = extract_min_img(instance)
        if bounded_img_np is None:
            # print("could not generate bounded box for image #", i)
            continue
        
        # Generate new images with translations of the original
        target_shape = instance_np.shape
        bounded_shape = bounded_img_np.shape
        max_dim0_offset = target_shape[0] - bounded_shape[0]
        max_dim1_offset = target_shape[1] - bounded_shape[1]
        max_offsets = [max_dim0_offset, max_dim1_offset]
        
        y_true = label.repeat(max_dim0_offset * max_dim1_offset)
        
        all_offsets_for_X = generate_all_offsets(bounded_img_np, instance_np, max_offsets)
        all_offsets_for_X = np_to_pytorch_imgs(all_offsets_for_X)
        
        new_inputs.append(all_offsets_for_X)
        new_targets.append(y_true)
        
    new_inputs = torch.cat(new_inputs, 0)
    new_targets = torch.cat(new_targets, 0)
    
    return new_inputs, new_targets

##########################################################################
## Feature Permutation ###################################################
##########################################################################
def permute_features(inputs, targets):
    '''
    Randomly permutes the column order of tabular data.
    No need to permute targets, just return as is...
    this is to fit into current augmentor architecture.
    '''
    inputs_dim = inputs.shape
    inputs = torch.flatten(inputs, start_dim=1)
    m, d = inputs.shape
    new_idx = torch.randperm(d)
    return inputs[:, new_idx].view(inputs_dim), targets

def permute_loaders(train_loader, test_loader, idx=None):
    '''
    Ensures that both train and test loaders are using 
    indices for fair comparison.
    '''
    X_train, y_train = train_loader.dataset.tensors
    X_test, y_test = test_loader.dataset.tensors
    X_train_dim = X_train.shape
    X_test_dim = X_test.shape
    X_train = torch.flatten(X_train, start_dim=1)
    X_test = torch.flatten(X_test, start_dim=1)
    m, d = X_train.shape
    if idx is None:
        idx = torch.randperm(d)
    X_train = X_train[:, idx].view(X_train_dim)
    X_test = X_test[:, idx].view(X_test_dim)
    trainer = torch.utils.data.TensorDataset(X_train, y_train)
    tester  = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(trainer, batch_size=train_loader.batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(tester, batch_size=test_loader.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader