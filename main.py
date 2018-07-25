# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(0), num_worker=4, accurate_landmark=False)


img = cv2.imread('/home/kits-adm/Datasets/flickr_epa/pics/258/000146.jpg')

# run detector
results = detector.detect_face(img)

if results is not None:

    total_boxes = results[0]
    points = results[1]
    
    # extract aligned face chips
    chips = detector.extract_image_chips(img, points, 144, 0.37)
    for i, chip in enumerate(chips):
        cv2.imshow('chip_'+str(i), chip)
        cv2.imwrite('chip_'+str(i)+'.png', chip)

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

    cv2.imshow("detection result", draw)
    cv2.waitKey(0)
