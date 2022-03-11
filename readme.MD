# coco dataset generator from mask images

### This is COCO label generator for binary(mask) images
After segmentation, if the image contains only one class, this is the effective tool to generate multiple training sample for deep learning model. (ex. yolo) 

## 1. Prepare your binary mask image

input \
  |-- image1.jpg \
  |-- image2.jpg \
  |-- ... 

## 2. run yolo_auto_gen.py
`python yolo_auto_gen.py --input {input mask dir} --putput {output save dir} --type {class}`

output \
  |-- image1.txt \
  |-- image2.txt \
  |-- ... 


![plot](./demo/yolo_gen.jpg)

If you have multiple classes in an image, you should manually change the class in the text.