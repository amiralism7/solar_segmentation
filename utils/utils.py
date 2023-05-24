import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tqdm import tqdm
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Model,load_model
from keras.applications import vgg16
# import keras_cv
from keras.layers import Input, Conv2D, Conv2DTranspose,\
                LeakyReLU, concatenate,\
                BatchNormalization, Dropout     
from keras.activations import sigmoid
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from numba import jit
from scipy.ndimage import gaussian_filter
from scipy.ndimage import find_objects, binary_fill_holes
from scipy.ndimage import generate_binary_structure, label
from scipy.optimize import linear_sum_assignment

IMG_SIZE = 372
best_threshold = 0.477

def read_data_path(input_dirs, target_dirs):
    """
    Gets the paths to all PNG images in the specified input directories and all PNG masks in the specified target directories.

    Args:
    input_dirs: A list of strings that refer to directories containing images.
    target_dirs: A list of strings that refer to directories containing masks.

    Returns:
    imgs_paths: A list of strings that refer to the paths of the images.
    masks_paths: A list of strings that refer to the paths of the masks.
    """
    imgs_paths = []
    masks_paths = []

    for input_dir, target_dir in zip(input_dirs, target_dirs):
        imgs_paths.extend([
            os.path.join(input_dir, fname)
            for fname in tqdm(os.listdir(input_dir)) 
            if fname.endswith(".png")
            ])
        masks_paths.extend([
            os.path.join(target_dir, fname)
            for fname in tqdm(os.listdir(target_dir)) 
            if fname.endswith(".png")
            ])
    return imgs_paths, masks_paths

def read_image_from_path(path):
    """
    Decodes an image file and returns a NumPy array.

    Args:
        path: A string that refers to the path of the image file.

    Returns:
        image: A NumPy array that contains the decoded image.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    return image[:,:,:3]

def show_images_in_row(images, titles=None):
    """
    Shows a list of images in a row.

    Args:
        images: A list of RGB images to be visualized.
        titles: A list of titles corresponding to the images.

    """
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        if n > 1:
            axs[i].axis("off")
            axs[i].imshow(images[i])
            if titles != None:
                axs[i].set_title(titles[i])
        else:
            axs.axis("off")
            axs.imshow(images[i])
            if titles != None:
                axs.set_title(titles[i])

def highlight_mask_on_image(image, mask):
    """
    Shows the given mask on the given image.

    Args:
        image: A NumPy array that stores values of a RGB image.
        mask: A NumPy array that stores values of a mask corresponding to the given image.

    Returns:
        image: A NumPy array that stores values of a RGB image, showing mask on the image with red color.
    """
    red_image = np.where(
    image[:,:,0:1] > mask, image[:,:,0:1], mask
    )
    image = np.concatenate([red_image, image[:,:,1:3]], axis=2)
    return image

def find_images_with_mask(img_paths, mask_paths):
    """
    finds the paths to all input images and target images that have a corresponding image in the other list.

    Args:
        img_paths: A list of strings that refer to the paths of the input images.
        mask_paths: A list of strings that refer to the paths of the target images.

    Returns:
        available_img_paths: A list of strings that refer to the paths of the input images that have a corresponding target image.
        available_mask_paths: A list of strings that refer to the paths of the target images that have a corresponding input image.
    """
    _mask_paths = set(mask_paths) # to have O(1) access
    available_img_paths = []
    available_mask_paths = []
    for img_path in img_paths:
        _mask_path = img_path.replace("/img/", "/mask/")
        if _mask_path in _mask_paths:
            available_img_paths.append(img_path)
            available_mask_paths.append(_mask_path)
    return available_img_paths, available_mask_paths


def create_raw_data_loader(X, Y):
    """
    Gets a `tf._ZipDataset` that passes a tuple which is a pair of an image and a mask.

    Args:
        X: A list of strings that refer to the paths of the input images.
        Y: A list of strings that refer to the paths of the target images.

    Returns:
        dataset: A `tf._ZipDataset` that contains pairs of images and masks.
    """
    
    def _shuffle_with_numpy(X, Y):
        indexes = np.random.permutation(len(X))
        _X = np.array(X)[indexes]
        _Y = np.array(Y)[indexes]
        return _X, _Y
    
    def _read_image(x_path, y_path):
        if x_path == y_path:
            X = read_image_from_path(x_path)
            Y = tf.zeros((400, 400, 1), dtype=tf.uint8)
        else:
            X = read_image_from_path(x_path)
            Y = read_image_from_path(y_path)
        return (X, Y)
    
    X, Y = _shuffle_with_numpy(X, Y)

    X = tf.data.Dataset.from_tensor_slices(X)
    Y = tf.data.Dataset.from_tensor_slices(Y)
    
    X_Y = tf.data.Dataset.zip((X, Y))
    X_Y = X_Y.map(_read_image)
    
    return X_Y


def normalize_resize(input_image, lbl=None):
    """
    Normalizes an image and a mask.

    Args:
        input_image: A NumPy array that stores values of a RGB image.
        lbl: A NumPy array that stores values of a mask.

    Returns:
        input_image: A NumPy array that stores values the normalized version of input_image.
        lbl: A NumPy array that stores values the normalized version of lbl.
    """
    if lbl==None:
        lbl = input_image
    input_image, lbl = input_image/255, lbl/255
    input_image = tf.image.resize(input_image, [IMG_SIZE, IMG_SIZE])
    lbl = tf.image.resize(lbl, [IMG_SIZE, IMG_SIZE])
    return input_image, lbl

def augment(input_image, lbl, IMG_SIZE=IMG_SIZE):
    """
    Augments an image and a mask.

    Args:
        input_image: A NumPy array that stores values of a RGB image.
        lbl: A NumPy array that stores values of a mask.

    Returns:
        input_image: The augmented version of input_image.
        lbl: The augmented version of lbl.
    """
    input_image = tf.image.resize(input_image, [IMG_SIZE, IMG_SIZE])
    lbl = tf.image.resize(lbl, [IMG_SIZE, IMG_SIZE])
    if tf.random.uniform(()) > 0.5:
        scale = np.random.uniform(0.90, 1)
        # use original image to preserve high resolution
        input_image = tf.image.central_crop(input_image, scale)
        lbl = tf.image.central_crop(lbl, scale)
        # resize
        input_image = tf.image.resize(input_image, (IMG_SIZE, IMG_SIZE))
        lbl = tf.image.resize(lbl, (IMG_SIZE, IMG_SIZE))

    # random brightness adjustment illumination
    input_image = tf.image.random_brightness(input_image, 0.3)
    # random contrast adjustment
    input_image = tf.image.random_contrast(input_image, 0.2, 0.5)
    # random saturation adjustment
    input_image = tf.image.adjust_saturation(
        input_image,
        tf.random.uniform([], minval=0.7, maxval=2)
    )


    # flipping random horizontal or vertical
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        lbl = tf.image.flip_left_right(lbl)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        lbl = tf.image.flip_up_down(lbl)
    input_image, lbl = input_image/255, lbl/255

    
    return input_image, lbl


def tf_dataset(dataset, batch_size=4, train=True, repeat=True):
    """
    Creates a training/validation/test dataloader.

    Args:
        dataset: A `tf.data.Dataset` object to go through some transformations.
        batch_size: An integer that specifies the size of the output batch of the transformed dataset object.
        train: A bool value which specifies if the dataset is going to be used for training or not.
        repeat: A bool value which specifies if the dataset is going to be repeated or not.

    Returns:
        dataset: A `tf.data.Dataset` object that has gone through some transformation and has infinite length.
    """
    if train:
        dataset = dataset.map(augment)
        # dataset = dataset.shuffle(20000)
    else:
        dataset = dataset.map(normalize_resize)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

def unet_model_creator(output_channels=1, weights="./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", IMG_SIZE=IMG_SIZE):
    """Creates a U-Net model for semantic segmentation tasks.

    Args:
      output_channels: The number of output channels (label channels).
      weights: The weights to load. Can be either "imagenet" to load the pre-trained ImageNet weights,
        or the path to a custom weights file.

    Returns:
      A `tf.keras.Model` that can be used for semantic segmentation tasks.

    """
    VGG16 = vgg16.VGG16(include_top=False, weights=weights, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model = VGG16.output
    
    for layer in VGG16.layers:
        layer.trainable = False   
        
    model = BatchNormalization()(model)
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)

    model = Conv2DTranspose(256,(3,3),strides=(2, 2))(model)
    model = LeakyReLU(0.1)(model)
    
    concat_1 = concatenate([model,VGG16.get_layer("block5_conv3").output])
    
    model = Conv2D(512,(3,3),strides=(1, 1), padding='same')(concat_1)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)    
    model = Conv2D(512,(3,3),strides=(1, 1), padding='same')(model)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)
    
    model = Conv2DTranspose(512,(3,3),strides=(2, 2),padding='same')(model)
    model = LeakyReLU(0.1)(model)
     
    
    concat_2 = concatenate([model,VGG16.get_layer("block4_conv3").output])
    
    model = Conv2D(512,(3,3),strides=(1, 1),padding='same')(concat_2)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)
    model = Conv2D(512,(3,3),strides=(1, 1),padding='same')(model)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)
    
    model = Conv2DTranspose(512,(3,3),strides=(2, 2))(model)
    model = LeakyReLU(0.1)(model)
    
    concat_3 = concatenate([model,VGG16.get_layer("block3_conv3").output])
    
    model = Conv2D(256,(3,3),strides=(1, 1),padding='same')(concat_3)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)
    model = Conv2D(256,(3,3),strides=(1, 1),padding='same')(model)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)
    
    
    model = Conv2DTranspose(256,(3,3),strides=(2, 2),padding='same')(model)
    model = LeakyReLU(0.1)(model)
    
    
    concat_4 = concatenate([model,VGG16.get_layer("block2_conv2").output])
    
    model = Conv2D(128,(3,3),strides=(1, 1),padding='same')(concat_4)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)
    model = Conv2D(128,(3,3),strides=(1, 1),padding='same')(model)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Dropout(0.2)(model)
    
    model = Conv2DTranspose(128,(3,3),strides=(2, 2),padding='same')(model)
    model = LeakyReLU(0.1)(model)
    
    concat_5 = concatenate([model,VGG16.get_layer("block1_conv2").output])
    
    model = Conv2D(64,(3,3),strides=(1, 1),padding='same')(concat_5)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)
    model = Conv2D(32,(3,3),strides=(1, 1),padding='same')(model)
    model = BatchNormalization()(model)  
    model = LeakyReLU(0.1)(model)

    model = Conv2D(output_channels,(3,3),strides=(1, 1),padding='same')(model)  
    model = sigmoid(model)
    
    model = Model(VGG16.input,model)
    return model

def fine_tune(unet_model, train=True, checkpoint_path=None, history_path='history.pickle', **kwargs):
    """Trains or loads a pre-trained U-Net model.

      Args:
        unet_model: A compiled U-Net model of type `tf.keras.Model`.
        train: A bool value that specifies whether we want to train it or load pretrained weights.
        checkpoint_path: Path to pretrained weights, needed when we want to load weights.
        history_path: Path to a pickled history of a trained model, needed when we want to load weights.
        **kwargs: Specifies arguments for when we are training the model.

      Returns:
        The unet_model with trained weights, of type `tf.keras.Model`.
        The history of the trained model, of type Dictionary.

      """
    if train:
        history = unet_model.fit(
            **kwargs
        )
        with open(history_path, 'wb') as handle:
            pickle.dump(history.history, handle)
        history = history.history
    else:
        unet_model.load_weights(checkpoint_path)
        
        try:
            with open(history_path, 'rb') as handle:
                history = pickle.load(handle)
        except:
            print("No loss/accuracy history for the model found.")
            history = None
    return unet_model, history


def fill_holes_and_remove_small_masks(masks, min_size=15, threshold=None):
    """ 
    fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    Args:

      masks: int, 2D or 3D array
          labelled masks, 0=NO masks; 1,2,...=mask labels,
          size [Ly x Lx] or [Lz x Ly x Lx]
      min_size: int (optional, default 15)
          minimum number of pixels per mask, can turn off with -1

    Returns:

      masks: int, 2D or 3D array
          masks with holes filled and masks smaller than min_size removed,
          0=NO masks; 1,2,...=mask labels,
          size [Ly x Lx] or [Lz x Ly x Lx]
    """
    if threshold:
        masks = masks > threshold
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None: 
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j + 1)
                j += 1
    return masks


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ 
    average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Args:

    masks_true: list of ND-arrays (int)
      where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int)
      ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns:

    ap: array [len(masks_true) x len(threshold)]
      average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
      number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
      number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
      number of false negatives at thresholds
    """
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n] + 0.00001)

    return ap, tp, fp, fn


@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Args:

    x: ND-array, int
      where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
      where 0=NO masks; 1,2... are mask labels

    Returns:

    overlap: ND-array, int
      matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs

    Args:

    masks_true: ND-array, int
      ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
      predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns:

    iou: ND-array, float
      matrix of IOU pairs of size [x.max()+1, y.max()+1]
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0

    return iou

def _true_positive(iou, th):
    """ true positive at threshold th

    Args:

    iou: float, ND-array
      array of IOU pairs
    th: float
      threshold on IOU for positive label

    Returns:

    tp: float
      number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()

    return tp

def get_masks_unet(output, threshold=0, min_size=30):
    """Creates masks using probability and assigning each "disconnected" part a number.

    Args:
    output: Output of our model, a Numpy ndarray of dtype np.float64.
    threshold: The threshold used to label pixels with classes, a float number.
    min_size: Minimum number of pixels in the masks, an integer.

    Returns:
    A masks assigning each "disconnected" part a number. A numpy array.

    """

    panels = output > threshold
    selem = generate_binary_structure(panels.ndim, connectivity=1)
    masks, nlabels = label(panels, selem)
    shape0 = masks.shape
    _,masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape0)
    masks = fill_holes_and_remove_small_masks(masks, min_size=min_size)

    return masks.astype(np.uint16)

def calc_precision(predicts, masks, threshold):
    """Calculates scores from a model's output, ground truth mask, and a threshold.

    Args:
    predicts: Output of our model, a Numpy ndarray of dtype np.float64.
    masks: The ground truth label, a `tf.tensor` of dtype tf.float64.
    threshold: The threshold used to label pixels with classes, a float number.

    Returns:
    The custom score of our prediction, labeled using threshold.
    A list that contains number of True_positives, false_positives and false_negative in the given batch respectively.

    """
    masks_generated = [get_masks_unet(lbl, threshold=threshold) for lbl in predicts]
    IoU_threshold = 0.5
    iou_threshold = np.array([IoU_threshold], "float64")
    ap = average_precision(masks.numpy().astype(int), masks_generated, threshold=iou_threshold)
    tp_fp_fn = [np.array(ap[1]).sum(), np.array(ap[2]).sum(), np.array(ap[3]).sum()]
    return np.array(ap[0]).mean(), tp_fp_fn

def find_best_threshold_batch_to_show(predicts, masks):
    """Calculates precision scores on each batch for a set of thresholds.

    Args:
    predicts: Output of our model, a numpy array of dtype np.float64.
    masks: The ground truth label, a `tf.tensor` of dtype tf.float64.

    Returns:
    A list of the thresholds (float numbers) used to label pixels with classes.
    A list of scores (float numbers) corresponding to thresholds.

    """
    map_threshold = np.linspace(0, 1, num=20, endpoint=True, dtype="float32")
    y_axis_ct = []
    for index, threshold in enumerate(map_threshold):
        cp = calc_precision(predicts, masks, threshold)
        y_axis_ct.append(cp[0])
        pass
    return map_threshold ,y_axis_ct 

def find_best_threshold_batch(predicts, masks):
    """Calculates precision scores on each batch for a set of thresholds.

    Args:
    predicts: Output of our model, a numpy array of dtype np.float64.
    masks: The ground truth label, a `tf.tensor` of dtype tf.float64.

    Returns:
    The best threshold for that batch, a float number.

    """
    map_threshold, y_axis_ct = find_best_threshold_batch_to_show(predicts, masks)
    return np.sum(np.array(y_axis_ct) * map_threshold)/np.sum(np.array(y_axis_ct))

def find_best_threshold(X_val, y_val, unet_model):
    """Finds the best threshold for a dataset.

    Args:
    X_val: list of paths to input images, a list of strings.
    y_val: list of paths to mask, a list of strings.
    unet_model: A tf.keras.Model that can be used for semantic segmentation tasks. 

    Returns:
    The best threshold for that dataset, a float number.

    """
    # our validation set is large, so we use a portion of it.
    ds_val = create_raw_data_loader(X_val[::7], y_val[::7])
    ds_val = tf_dataset(ds_val, batch_size=8, train=False, repeat=False)
    thresholds = []
    for x, y in tqdm(ds_val):
        predicted = unet_model.predict(x, verbose=0)
        thr = find_best_threshold_batch(predicted, y)
        thresholds.append(thr)
    return np.nanmean(np.array(thresholds))

def calc_precision_test(X_test, y_test, unet_model, threshold=best_threshold):
    """Calculates some score on a dataset given a threshold.

    Args:
    X_test: List of paths to input images, a list of strings.
    y_test: List of paths to mask, a list of strings.
    unet_model: A tf.keras.Model that can be used for semantic segmentation tasks. 
    threshold: The threshold to be used, a float number.

    Returns:
    A dictionary containing scores and the batch with lowest score.

    """
    min_precision = 0.99999
    batch_min_precision = None
    
    ds_test = create_raw_data_loader(X_test, y_test)
    ds_test = tf_dataset(ds_test, batch_size=8, train=False, repeat=False)
    precisions = []
    true_positives = []
    false_positives = []
    false_negatives = []
    for x, y in tqdm(ds_test):
        predicted = unet_model.predict(x, verbose=0)
        # calculate scores on each batch
        calc = calc_precision(predicted, y, threshold)
        precision = calc[0]
        true_positive, false_positive, false_negative = calc[1]
        precisions.append(precision)
        true_positives.append(true_positive)
        false_positives.append(false_positive)
        false_negatives.append(false_negative)
        
        if precision < min_precision:
            min_precision = precision
            batch_min_precision = (x, y)
    output_dict = {
        "average_precision_custom" : np.mean(np.array(precisions)),
        "sum_true_positive" : np.sum(np.array(true_positives)),
        "sum_false_positive" : np.sum(np.array(false_positives)),
        "sum_false_negative" : np.sum(np.array(false_negatives)),
        "min_precision_batch" : min_precision,
        "batch_with_min_precision": batch_min_precision,
    }    
    return output_dict


def draw_box_single(image, mask, seg_value=1):
    """Draws a box around the object corresponding to seg_value in mask.

    Args:
    image: A NumPy array that stores values of a RGB image.
    mask: A NumPy array that stores values of a mask.
    seg_value: An integer that specifies the pixel values we want to draw box around.

    Returns:
    A NumPy array that contains the value of image and a box around the object corresponding to seg_value and mask.

    """
    image = np.array(image)

    if mask is not None:
        np_seg = np.array(mask)
        segmentation = np.where(np_seg == seg_value)

        # Bounding Box
        bbox = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            start_point = (x_min, y_min)
            end_point = (x_max, y_max)
            
            color = [0, 0, 0]
            color[seg_value % 3] = 255
            if np.max(image) > 1:
                image_box = cv2.rectangle(image , start_point, end_point, color, 2) / 255
            else:
                image_box = cv2.rectangle(image * 255, start_point, end_point, color, 2) / 255
            return image_box
        else: 
            return image
    else:
        # Handle error case where segmentation image cannot be read or is empty
        print("Error: Segmentation image could not be read or is empty.")
        return "Error"
    


def draw_box_all(image, mask):
    """Draws a box around the all objects marked in the mask.

    Args:
    image: A NumPy array that stores values of a RGB image.
    mask: A NumPy array that stores values of a mask.

    Returns:
    A NumPy array that contains the value of image and a box around all objects marked in mask.

    """
    num_box = np.max(mask)
    for i in range(1, num_box + 1):
        image = draw_box_single(image, mask, seg_value=i)
    return image




def detect_panels(images, unet_model,threshold=best_threshold, show=False):
    """Detects panels in an image using a U-Net model.

    Args:
        image: A NumPy array that stores values of a RGB image.
        unet_model: A tf.keras.Model that can be used for semantic segmentation tasks.
        threshold: A float value (or "adaptive") as the threshold to be used to label pixels with classes.
        show: A bool that specifies whether we want to show the output images or not.

    Returns:
        A NumPy array that contains the value of image and a box around the objects found by the model.

    """
    
    def _detect_panels_single(image, predicted, threshold=best_threshold, show=False):
        predicted_mask_colored = get_masks_unet(predicted, threshold=threshold)
        image_with_box = draw_box_all(image, predicted_mask_colored)
        titles = ["Image", "Predicted Mask", "With Detection Boxes", "Model's Output"]
        if show:
            show_images_in_row([image, predicted_mask_colored, image_with_box, predicted], titles)
        return image_with_box

    if len(images.shape) == 3:
        images = tf.reshape(images, (1, *images.shape))
    images = normalize_resize(images)[0]
    
    predicted = unet_model(images)
    if threshold == "adaptive":
        threshold = min(np.max(predicted) / 3, 0.1)

    outputs = []
    for i in range(len(images)):
        out = _detect_panels_single(images[i], predicted[i], threshold=threshold, show=show)
        outputs.append(out)
    return np.array(outputs)



def _test_path(path):
    def _check_url(string):
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        url = re.findall(regex, string)
        return [x[0] for x in url]
    def _download_image(path):
        fname = "image"
        try:
            r = requests.get(path, stream = True)
        except requests.ConnectionError:
            print("Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("Failed to download data !!!")
            else:
                with open("./" + fname, "wb") as fid:
                    for block in r.iter_content(chunk_size = 1024):
                        if block: 
                            fid.write(block)
    #####################################
    try:
        path == _check_url(path)[0]
        _download_image(path)
        image = read_image_from_path("image")
    except IndexError:
        try:
            image = read_image_from_path(path)
        except:
            print("Not Found!")
            image = None

    return image
        
    
def test_model(image, path_model, threshold=best_threshold):
    """
    Segments the image using our pretrained model.

    Args:
    
    image: A NumPy array (or a path/link to an image) that stores values of a RGB image. (or any object that can be casted to a Numpy array)
    path_model: path to our models weights
    threshold: A float value as the threshold to be used to label pixels with classes, a float number. or it can be set to "adaptive", useful for cases that the input image comes from a different distribution than the training set.

    Returns:
    None. The results are printed to the console.

    """

    def _test_path(path):
        def _check_url(string):
            regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
            url = re.findall(regex, string)
            return [x[0] for x in url]
        def _download_image(path):
            fname = "image"
            try:
                r = requests.get(path, stream = True)
            except requests.ConnectionError:
                print("Failed to download data !!!")
            else:
                if r.status_code != requests.codes.ok:
                    print("Failed to download data !!!")
                else:
                    with open("./" + fname, "wb") as fid:
                        for block in r.iter_content(chunk_size = 1024):
                            if block: 
                                fid.write(block)
        #####################################
        try:
            path == _check_url(path)[0]
            _download_image(path)
            image = read_image_from_path("image")
        except IndexError:
            try:
                image = read_image_from_path(path)
            except:
                print("Not Found!")
                image = None

        return image
        
    unet_model = unet_model_creator(weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
    unet_model.compile(
        optimizer = Adam(learning_rate = 1e-4), 
        loss = 'binary_crossentropy', 
        metrics = ['accuracy']
    )
    unet_model, _ = fine_tune(
        unet_model,
        train=False,
        checkpoint_path=path_model,
        history_path='./history.pickle',
    )
    
    check_next_cond = True
    
    if type(image) == str:
        image = _test_path(image)
    list_images = None
    
    if type(image) == list and type(image[0]) == str:
        list_images = [_test_path(path) for path in image]
        image == None
        check_next_cond = False
            
    if check_next_cond and image != None:
        if len(image.shape) == 3:
            image = tf.reshape(image, (1, *image.shape))
        detect_panels(image, unet_model, threshold, True)  
    
    if list_images != None:
        for image in list_images:
            if len(image.shape) == 3:
                image = tf.reshape(image, (1, *image.shape))
            detect_panels(image, unet_model, threshold, True) 

