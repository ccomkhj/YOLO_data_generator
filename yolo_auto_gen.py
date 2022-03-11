import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from loguru import logger

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Crop image generator")

    parser.add_argument("--input",
                        default='input/lettuce',
                        help="Location of input image directory to be cropped.")

    parser.add_argument("--output",
                        default='output',
                        help="Location of output directory to save cropped images.")

    parser.add_argument("--type",
                        default='lettuce',
                        help="Types of the cropped image.")

    args = parser.parse_args()
    return args

def group(mask, sensitivity, debug=False, intact=True):
    '''
    group independent objects.

    Input
    mask: location of segmentation mask (after segmentation process)
    sensitivity : this ratio is the parameter to decide, based on the object size (recommend: 0.0001 ~ 0.01)
    intact: keep it True, if you want to ignore object at the border

    Output
    cropped_masks: a list of cropped object masks
    roi: region of interest for each object mask

    '''
    
    img = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel=(3,3))

    sketch = img.copy()
    num_groups, _, bboxes, centers  = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    logger.info(f'The number of {num_groups-1} objects are detected.')
    # bboxes: left most x, top most y, horizontal size, vertical size, total area 
    
    MIN_AREA = img.shape[0] * img.shape[1] * sensitivity 

    cropped_masks = []
    rois = []

    for i, (bbox, center) in enumerate(zip(bboxes, centers)):
        tx, ty, hori, verti, area = bbox
        if area < MIN_AREA or i == 0:
            # skip too small or the whole image
            continue
        roi = ty, ty+verti, tx, tx+hori

        cropped = img[roi[0]:roi[1], roi[2]:roi[3]]

        if cv2.connectedComponents(cropped)[0] != 2: # if there is more than one object in the cropped mask,
            continue

        if intact and any(x in roi for x in [0, img.shape[0], img.shape[1]]): # if a cropped image is located on the image border,
            continue

        cropped_masks.append(cropped)
        rois.append(roi)

        if debug:
            sketch = cv2.rectangle(sketch, pt1= (tx, ty), pt2= (tx+hori, ty+verti), color= 1, thickness= 3)
            f, axarr = plt.subplots(2)
            axarr[0].imshow(sketch)
            axarr[1].imshow(cropped)
            plt.show()


    logger.info(f'The number of {len(cropped_masks)} objects are saved.')

    return cropped_masks, rois

def roi2coco(roi, size):
    '''
    convert roi to COCO format.

    Input
    roi: region of interest for each object mask; y left top, y right bottom, x left top, x right bottom.
    size: shape of image; y size, x size

    Output
    coco: normalized xywh format (from 0 to 1)
    '''
    center_x = round((roi[2]+roi[3])/2/size[1], 4)
    center_y = round((roi[0]+roi[1])/2/size[0], 4)
    w_ratio = round((-roi[2]+roi[3])/size[1], 4)
    h_ratio = round((-roi[0]+roi[1])/size[0], 4)
    return center_x, center_y, w_ratio, h_ratio

def generate(type, center_x, center_y, w_ratio, h_ratio, txt_path, CLASS):

    with open(txt_path + '.txt', 'a') as f:
        f.write(('%g ' * 5 + '\n') % (CLASS.index(type), center_x, center_y, w_ratio, h_ratio)) 

if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input)

    CLASS = ['lettuce', 'basil']

    mask_files = list(sorted(input_dir.glob("*.png" or "*.jpg" or "*.jpeg")))
    logger.info(f"Found {len(mask_files)} mask(s) in {args.input}:\n{mask_files}")
    if not mask_files:
        exit()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {output_dir}")

    for mask_file in mask_files:
        mask = cv2.cvtColor(cv2.imread(mask_file.__str__()), cv2.COLOR_BGR2GRAY)
        _, rois = group(mask, sensitivity=0.0002, debug=False)
        
        txt_name = os.path.join(output_dir,mask_file.stem)

        for roi in rois:

            center_x, center_y, w_ratio, h_ratio = roi2coco(roi, mask.shape)
            generate(args.type, center_x, center_y, w_ratio, h_ratio, txt_name, CLASS)
