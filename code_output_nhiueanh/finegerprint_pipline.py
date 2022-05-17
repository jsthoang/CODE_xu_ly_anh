import cv2 as cv
from glob import glob
import os
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize

ip_img = cv.imread("F:/CODE/code_output_nhiueanh/segmented.jpg",0)

def fingerprint_pipline(input_img):
    block_size = 16

    # pipe line picture re https://www.cse.iitk.ac.in/users/biometrics/pages/111.JPG
    # normalization -> orientation -> frequency -> mask -> filtering

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # color threshold
    _, threshold_img = cv.threshold(normalized_img,0,255,cv.THRESH_BINARY| cv.THRESH_OTSU)
    cv.imshow('color_threshold', threshold_img); cv.waitKeyEx()
    normalized_img = threshold_img

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    #cv.imwrite('segmented.jpg', segmented_img)

    

    # orientations
    #angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    #orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    #cv.imshow('ori', orientation_img)
    #gabor_img = segmented_img
    # find the overall frequency of ridges in Wavelet Domain
    #freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    # create gabor filter and do the actual filtering
    #gabor_img = gabor_filter(normim, angles, freq)
    #gabor_img = normalized_img

    # thinning oor skeletonize
    #them//////
    mty = np.zeros(ip_img.shape, np.uint8) 
    ip2_img = cv.bitwise_or(mty, ip_img)
    cv.imshow('ip2', ip2_img)
    #het them
    thin_image = skeletonize(ip2_img)

    # minutias
    minutias = calculate_minutiaes(thin_image)

    # singularities
    #singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)

    # visualize pipeline stage by stage
    output_imgs = [input_img, normalized_img, segmented_img, thin_image, minutias]
    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    #results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)
    results = minutias
    
    return results


if __name__ == '__main__':
    # open images
    img_dir = './DATAbase/*'
    #img_dir = "F:/CODE/code_output_nhiueanh/sample_tu_tao"
    output_dir = './output/'
    def open_images(directory):
        images_paths = glob(directory)
        return np.array([cv.imread(img_path,0) for img_path in images_paths])

    images = open_images(img_dir)

    # image pipeline
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(tqdm(images)):
        results = fingerprint_pipline(img)
        cv.imwrite(output_dir+str(i)+'.png', results)
        cv.imshow('image pipeline', results); cv.waitKeyEx()
