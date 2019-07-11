# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import sys

sys.path.append('../')
import pose_utils as pu

# construct the argument parse and parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
'''



def runFaceDetect(frame, net, factor=0.25, _alexNetSize=227, conf_thr=0.5):
    # grab the frame dimensions and convert it to a blob
    cropped_img = frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()
    max_conf_idx = np.argmax(detections[0,0,:,2])
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, max_conf_idx, 2]
    #print(confidence)
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    #if confidence < args["confidence"]:
    if confidence < conf_thr:
        return (-1, cropped_img)
        #return -1

    # compute the (x, y)-coordinates of the bounding box for the
    # object
    box = detections[0, 0, max_conf_idx, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    #print((startX, startY, endX, endY))
    width = endX - startX
    height = endY - startY
    bbox_dict = {}
    #print((startX, startY, width, height))
    bbox_dict['x'] = startX
    bbox_dict['y'] = startY
    bbox_dict['width'] = width
    bbox_dict['height'] = height
    #print(bbox_dict)
    cropped_img = pu.cropFaceImage(frame, bbox_dict, factor, _alexNetSize)        
    #print(type(cropped_img))
    #print(_cropped_img.shape)

    # draw the bounding box of the face along with the associated
    # probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10        
    cv2.rectangle(frame, (startX, startY), (endX, endY),
        (0, 0, 255), 2)		
    cv2.putText(frame, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return (1, cropped_img)
    
if __name__ == '__main__':
    prototxt = './deploy.prototxt'
    model = './res10_300x300_ssd_iter_140000.caffemodel'


    # load our serialized model from disk
    print("[INFO] loading model...")
    #net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        _, cropped_img = runFaceDetect(frame, net) # frame is updated by this function, showing a bbox of face        
        #print(cropped_img.shape)
        '''
        if type(cropped_img) == int:
            print('no faces')
            continue
        '''

        # show the output frame
        cv2.imshow("Frame", frame)
        cv2.imshow('Cropped Face', cropped_img)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()




