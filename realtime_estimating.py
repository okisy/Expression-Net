import numpy as np
import cv2
import time
import sys
import imutils
from imutils.video import VideoStream

sys.path.append('./FaceDetect')
from crop_face import runFaceDetect

# Assuming that this module and main.py are in the same directory.
prototxt = './FaceDetect/deploy.prototxt'
model_path = './FaceDetect/res10_300x300_ssd_iter_140000.caffemodel'
cv2net_info = (prototxt,model_path)

def realtime_estimating(sess, FLAGS, fc1le, pose_model, x, cv2net_info=cv2net_info):
    
    # load our serialized model from disk
    print("[INFO] loading face detection model...")
    net = cv2.dnn.readNetFromCaffe(cv2net_info[0], cv2net_info[-1])

    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    while True:
        start = time.time()
        
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        faceOrNot, image = runFaceDetect(frame, net)
        if faceOrNot == -1:
            print('no faces detected')
            cv2.imshow("Frame", frame)
            continue

        # image = cv2.imread('./tmp/subject10_a.jpg', 1)
        # image = np.asarray(image)

        # Fix the grey image
        if len(image.shape) < 3:
            image_r = np.reshape(image, (image.shape[0], image.shape[1], 1))
            image = np.append(image_r, image_r, axis=2)
            image = np.append(image, image_r, axis=2)

        image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
        (Expr, Pose) = sess.run([fc1le, pose_model.preds_unNormalized], feed_dict={x: image})

        Pose = np.reshape(Pose, [-1])
        Expr = np.reshape(Expr, [-1])

        print ('Pose coefficients:')
        print (Pose)
        print ('Expression coefficients:')
        print (Expr)

        print(time.time() - start)
        
        # show the output frame
        cv2.imshow("Frame", frame)
        # cv2.imshow('Cropped Face', image)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
'''
if __name__ == '__main__':    
    realtime_estimating()
'''