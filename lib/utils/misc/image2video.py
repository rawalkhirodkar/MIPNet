import cv2
import os
import numpy as np


# ----------------------------------------------
IMAGE_FOLDER = '/Users/khirawal/Desktop/datasets/coco_sorted_instance/visualize'
VIDEO_NAME = os.path.join(IMAGE_FOLDER, 'video.avi')

SPF = 2.0 #seconds spent on an image
SKIP_IMAGE = 1.0 #write every nth frame to video

# ----------------------------------------------
images = sorted([img for img in os.listdir(IMAGE_FOLDER) if (img.endswith(".png") or img.endswith(".jpg"))])

frame = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(VIDEO_NAME, fourcc, fps=1/SPF, frameSize=(width, height))

for i, image in enumerate(images):
	if i%SKIP_IMAGE == 0:
		print(os.path.join(IMAGE_FOLDER, image))
		video.write(cv2.imread(os.path.join(IMAGE_FOLDER, image)))


cv2.destroyAllWindows()
video.release()

# ----------------------------------------------
