# USAGE
# python yolo_live.py

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from imutils.video import VideoStream
import os
language = "en"
import tkinter as tk
import json

def send_email():
	import smtplib
	s = smtplib.SMTP('smtp.gmail.com', 587)
	s.starttls()
	s.login("sudeeshnapatel@gmail.com", "jirnwjwvczgjmdgj")
	message = "Crow Limit is Exceeded"
	s.sendmail("sudeeshnapatel@gmail.com", "sudeeshnapatel@gmail.com", message)
	s.quit()

is_mail_send=False

def objectdetection():

	global is_mail_send

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join(["yolo-coco", "obj.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# initialize a list of colors to represent each possible class label
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join(["yolo-coco", "yolo.weights"])
	configPath = os.path.sep.join(["yolo-coco", "yolo.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# start the video stream thread
	print("[INFO] starting video stream thread...")
	vs = cv2.VideoCapture("videos/1.mp4")
	# vs = cv2.VideoCapture(0)

	time.sleep(1.0)

	# loop over frames from the video stream
	while True:
		
		count_dict = {}

		(grabbed, image) = vs.read()

		if not grabbed:
			break

		(H, W) = image.shape[:2]

		# determine only the *output* layer names that we need from YOLO
		ln = net.getLayerNames()
		#ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > 0.5:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.3)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates

				if(classIDs[i]==0):
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					print("=======",classIDs[i],"==================")

					# draw a bounding box rectangle and label on the image
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					#speech = gTTS(text = text, lang = language, slow = False)
					#speech.save("text.mp3")
					lab=LABELS[classIDs[i]]
					if lab in list(count_dict.keys()):
						count_dict[lab]=count_dict[lab]+1
					else:
						count_dict.update({lab:1})

		result = json.dumps(count_dict)
		cv2.putText(image,result, (0,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
		# show the output image
		cv2.imshow("Image", image)
		#os.system("text.mp3")

		if count_dict["person"]>20:
			try:
				if not is_mail_send:
					print("Mail Sending")
					send_email()
					is_mail_send=True
				else:
					print("Mail allready Sent")
			except Exception as e:
				print("Mail Sending Failed")

		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

objectdetection()