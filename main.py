# import the necessary packages
from utils import config
from utils.utils import min_size, compute_perspective_transform, compute_point_perspective_transformation
from utils.model import ModelServer
from scipy.spatial import distance as dist
import numpy as np
import argparse
import cv2
import os

def get_ROI(width,height):
	# CAREFULLY DEFINE POINTS FOR THE ROI FOR CORRECT ALIGNMENT, VARIES WITH SOURCE VIDEO
	# Here I have taken the whole frame as ROI
	return [[0,0],[0,height],[width,height],[width,0]] # IMPORTANT!!!! 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
args = vars(ap.parse_args())

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vs.get(cv2.CAP_PROP_FPS))
if args["output"] != "" and writer is None:
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	writer = cv2.VideoWriter(args["output"], fourcc, fps, (width,height))

# Models
person_model = ModelServer('person')
face_detector = ModelServer('face')
mask_model = ModelServer('mask')

# Birds eye view transform
_,frame = vs.read()
corner_points = get_ROI(width,height) 
matrix,imgOutput = compute_perspective_transform(corner_points,width,height,frame)
out_shape = imgOutput.shape[:2]

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	classes, scores, boxes = person_model.detect(frame,config.MIN_CONF)
	results = []
	for i,bbox in enumerate(boxes):
		x = (bbox[0]+bbox[2])//2
		y = (bbox[1]+bbox[3])//2
		# Filter out far away people using minimum area threshold for bounding box. 
		# Distance calculation not very accurate at long distance.
		if min_size(bbox):
			results.append([bbox.tolist(),(x,y),(x,bbox[3])])

	# initialize the set of indexes that violate the minimum social
	# distance
	violate = set()

	new_img = np.zeros(imgOutput.shape)
	# ensure there are at least two people detected (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:

		centroids = np.array([r[1] for r in results])
		downoids = np.array([r[2] for r in results])
		transformed_downoids = compute_point_perspective_transformation(matrix,downoids)

		D = dist.cdist(transformed_downoids, transformed_downoids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels

				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)

	# Mask Violation code here
	_,_,faces = face_detector.detect(frame,0.7)
	mask_violations = 0
	if len(faces)>0:
		for face_bbox in faces:
			face_bbox[0],face_bbox[1]=face_bbox[0]-7,face_bbox[1]-7
			face_bbox[2],face_bbox[3]=face_bbox[2]+7,face_bbox[3]+7
			face_bbox = np.clip(face_bbox,0,np.inf).astype(np.int) # Handle -ve values for bbox.
			face_crop = frame[face_bbox[1]:face_bbox[3],face_bbox[0]:face_bbox[2]]
			pred = mask_model.check_mask(face_crop)
			if pred:
				face_color = (0,255,0) #GREEN, mask
			else:
				face_color = (0,0,255) #RED, no mask
				mask_violations +=1
			cv2.rectangle(frame,(face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]),face_color,1)
			
	text = "Mask Violations: {}".format(mask_violations)
	cv2.putText(frame, text, (10, frame.shape[0] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	# loop over the results
	for (i,(bbox, centroid, _)),downoids in zip(enumerate(results),transformed_downoids):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		(x, y) = downoids
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = (0, 0, 255)

		# draw the centroid coordinates of the person,
		# cv2.rectangle(frame,(bbox[0], bbox[1]), (bbox[2], bbox[3]),(0,0,255),1)
		cv2.circle(frame, (cX, cY), 3, color, -1)

		cv2.circle(new_img, (x, y), 20, color, 2)
		cv2.circle(new_img, (x, y), 5, color, -1)

	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 35),
		cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	# check to see if the output frame should be displayed to our
	# screen
	show_img = np.zeros((frame.shape[0],int(frame.shape[1]*1.5),frame.shape[2]))
	new_img = cv2.rotate(new_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
	new_img = cv2.flip(new_img,1)
	new_img = cv2.resize(new_img,(frame.shape[1]//2,frame.shape[0]))

	show_img[:,:frame.shape[1],:] = frame
	show_img[:,frame.shape[1]:,:] = new_img
	show_img = show_img.astype(np.uint8)

	# show the output frame
	show_img = cv2.resize(show_img,(1080,640)) # To fit in screen
	cv2.imshow('Social Distancing',show_img)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
			break

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		show_img = cv2.resize(show_img,(width,height))
		writer.write(show_img)