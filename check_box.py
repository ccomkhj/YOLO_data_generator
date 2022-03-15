import cv2
import os
import random
from pathlib import Path
from loguru import logger
import argparse



def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes to check the annotation.")

    parser.add_argument("--label",
                        default='output',
                        help="Location of label directory.")

    parser.add_argument("--image",
                        default='input',
                        help="Location of raw iamge directory to load for checking.")

    parser.add_argument("--save",
                        default='save_image',
                        help="Location of the directory for saving boundng boxed-output.")

    args = parser.parse_args()
    return args

def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_box_on_image(image_name, classes, colors, label_folder, raw_images_folder, save_images_folder ):
    image_name = image_name.stem
    txt_path  = os.path.join(label_folder,'%s.txt'%(image_name))  

    image_path = os.path.join( raw_images_folder,'%s.png'%(image_name)) 
    save_file_path = os.path.join(save_images_folder,'%s.png'%(image_name)) 
    try:
        source_file = open(txt_path)
    except:
        logger.info(f"no relevant label: {txt_path}")
        return 0
    image = cv2.imread(image_path)
    try:
        height, width, channels = image.shape
    except:
        logger.info('no shape info.')
        return 0

    box_number = 0
    for line in source_file: 
        staff = line.split() 
        class_idx = int(staff[0])

        x_center, y_center, w, h = float(staff[1])*width, float(staff[2])*height, float(staff[3])*width, float(staff[4])*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2)     

        plot_one_box([x1,y1,x2,y2], image, color=colors[class_idx], label=classes[class_idx], line_thickness=None)

        cv2.imwrite(save_file_path,image) 

        box_number += 1
    return box_number


if __name__ == '__main__':       
    args = parse_args()
    label_folder = args.label
    raw_images_folder = args.image
    save_images_folder = args.save

    CLASS = ['plant']
    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASS))]

    image_names = Path(raw_images_folder).glob("*")

    image_total = 0
    logger.info(f'{len(image_names)} images will be processed.')
    for image_name in image_names:
        box_num = draw_box_on_image(image_name, CLASS, colors, label_folder, raw_images_folder, save_images_folder)
        image_total += 1
        logger.info(f'{image_total}th image has {box_num} boxes:')
    
    logger.info(f'Drawing box process is successfully done!')