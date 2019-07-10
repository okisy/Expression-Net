import numpy as np
import cv2
import time


def realtime_estimating(sess, FLAGS, fc1le, pose_model, x):
    while True:
        start = time.time()
        image = cv2.imread('./tmp/subject10_a.jpg', 1)

        image = np.asarray(image)
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
