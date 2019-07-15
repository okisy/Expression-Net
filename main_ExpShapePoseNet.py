import sys
import imutils
import numpy as np
import tensorflow as tf
import cv2
import scipy.io as sio
from imutils.video import VideoStream
import os
import os.path
import time
import scipy
import ST_model_nonTrainable_AlexNetOnFaces as Pose_model
import utils
import open3d

from FaceDetect.crop_face import runFaceDetect

sys.path.append('./kaffe')
sys.path.append('./ResNet')
from ThreeDMM_shape import ResNet_101 as resnet101_shape
from ThreeDMM_expr import ResNet_101 as resnet101_expr

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 227, 'Image side length.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size')

# Global parameters
_tmpdir = './tmp/'  # save intermediate images needed to fed into ExpNet, ShapeNet, and PoseNet
print('> make dir')
if not os.path.exists(_tmpdir):
    os.makedirs(_tmpdir)
output_proc = 'output_preproc.csv'  # save intermediate image list
factor = 0.25  # expand the given face bounding box to fit in the DCCNs
_alexNetSize = 227
mesh_folder = './output_ply'  # The location where .ply files are saved

# Get training image/labels mean/std for pose CNN
file = np.load("./fpn_new_model/perturb_Oxford_train_imgs_mean.npz")
train_mean_vec = file["train_mean_vec"]  # [0,1]
del file
file = np.load("./fpn_new_model/perturb_Oxford_train_labels_mean_std.npz")
mean_labels = file["mean_labels"]
std_labels = file["std_labels"]
del file

# Get training image mean for Shape CNN
mean_image_shape = np.load('./Shape_Model/3DMM_shape_mean.npy')  # 3 x 224 x 224
mean_image_shape = np.transpose(mean_image_shape, [1, 2, 0])  # 224 x 224 x 3, [0,255]

# Get training image mean for Expression CNN
mean_image_exp = np.load('./Expression_Model/3DMM_expr_mean.npy')  # 3 x 224 x 224
mean_image_exp = np.transpose(mean_image_exp, [1, 2, 0])  # 224 x 224 x 3, [0,255]


def face_reconstruction():
    # placeholders for the batches
    x = tf.compat.v1.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])

    ###################
    # Face Pose-Net
    ###################
    start = time.time()

    net_data = np.load("./fpn_new_model/PAM_frontal_ALexNet_py3.npy").item() # 854MB
    pose_labels = np.zeros([FLAGS.batch_size, 6])
    x1 = tf.compat.v1.image.resize_bilinear(x, tf.constant([227, 227], dtype=tf.int32))

    # # Image normalization
    x1 = x1 / 255.  # from [0,255] to [0,1]
    # subtract training mean
    mean = tf.reshape(train_mean_vec, [1, 1, 1, 3])
    mean = tf.cast(mean, 'float32')
    x1 = x1 - mean

    pose_model = Pose_model.Pose_Estimation(x1, pose_labels, 'valid', 0, 1, 1, 0.01, net_data,
                                            FLAGS.batch_size, mean_labels, std_labels)
    pose_model._build_graph()
    del net_data

    print("Face Pose Net {}".format(time.time() - start))
    start = time.time()

    ###################
    # Shape CNN
    ###################
    x2 = tf.compat.v1.image.resize_bilinear(x, tf.constant([224, 224], dtype=tf.int32))
    x2 = tf.cast(x2, 'float32')
    x2 = tf.reshape(x2, [FLAGS.batch_size, 224, 224, 3])

    # Image normalization
    mean = tf.reshape(mean_image_shape, [1, 224, 224, 3])
    mean = tf.cast(mean, 'float32')
    x2 = x2 - mean

    with tf.variable_scope('shapeCNN'):
        net_shape = resnet101_shape({'input': x2}, trainable=True)
        pool5 = net_shape.layers['pool5']
        pool5 = tf.squeeze(pool5)
        pool5 = tf.reshape(pool5, [FLAGS.batch_size, -1])

        npzfile = np.load('./ResNet/ShapeNet_fc_weights.npz')
        ini_weights_shape = npzfile['ini_weights_shape']
        ini_biases_shape = npzfile['ini_biases_shape']
        with tf.variable_scope('shapeCNN_fc1'):
            fc1ws = tf.Variable(tf.reshape(ini_weights_shape, [2048, -1]), trainable=True, name='weights')
            fc1bs = tf.Variable(tf.reshape(ini_biases_shape, [-1]), trainable=True, name='biases')
            fc1ls = tf.nn.bias_add(tf.matmul(pool5, fc1ws), fc1bs)

    print("Shape CNN {}".format(time.time() - start))
    start = time.time()

    ###################
    # Expression CNN
    ###################
    with tf.compat.v1.variable_scope('exprCNN'):
        net_expr = resnet101_expr({'input': x2}, trainable=True)
        pool5 = net_expr.layers['pool5']
        pool5 = tf.squeeze(pool5)
        pool5 = tf.reshape(pool5, [FLAGS.batch_size, -1])

        npzfile = np.load('./ResNet/ExpNet_fc_weights.npz')
        ini_weights_expr = npzfile['ini_weights_expr']
        ini_biases_expr = npzfile['ini_biases_expr']
        with tf.compat.v1.variable_scope('exprCNN_fc1'):
            fc1we = tf.Variable(tf.reshape(ini_weights_expr, [2048, 29]), trainable=True, name='weights')
            fc1be = tf.Variable(tf.reshape(ini_biases_expr, [29]), trainable=True, name='biases')
            fc1le = tf.nn.bias_add(tf.matmul(pool5, fc1we), fc1be)

    # Add ops to save and restore all the variables.
    init_op = tf.compat.v1.global_variables_initializer()
    saver_pose = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           scope='Spatial_Transformer'))
    saver_ini_shape_net = tf.train.Saver(var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
    saver_ini_expr_net = tf.compat.v1.train.Saver(
        var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))

    print("Expression CNN {}".format(time.time() - start))

    ###################
    # Facial Detector
    ###################
    prototxt = './FaceDetect/deploy.prototxt'
    model_path = './FaceDetect/res10_300x300_ssd_iter_140000.caffemodel'
    cv2net_info = (prototxt, model_path)

    # load our serialized model from disk
    print("[INFO] loading face detection model...")
    net = cv2.dnn.readNetFromCaffe(cv2net_info[0], cv2net_info[-1])

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        start = time.time()
        sess.run(init_op)

        # Load face pose net model from Chang et al.'ICCVW17
        load_path = "./fpn_new_model/model_0_1.0_1.0_1e-07_1_16000.ckpt"
        saver_pose.restore(sess, load_path)
        print('> Loaded the Pose Model {}'.format(time.time() - start))
        start = time.time()

        # load 3dmm shape and texture model from Tran et al.' CVPR2017
        load_path = "./Shape_Model/ini_ShapeTextureNet_model.ckpt"
        saver_ini_shape_net.restore(sess, load_path)
        print('> Loaded the Shape Model Model {}'.format(time.time() - start))
        start = time.time()

        # load our expression net model
        load_path = "./Expression_Model/ini_exprNet_model.ckpt"
        saver_ini_expr_net.restore(sess, load_path)
        print('> Loaded the Expression Model {}'.format(time.time() - start))
        start = time.time()

        # Modified Basel Face Model
        BFM_path = './Shape_Model/BaselFaceModel_mod.mat' # 140MB
        model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
        model = model["BFM"]
        faces = model.faces - 1
        print('> Loaded the Basel Face Model to write the 3D output! {}'
              .format(time.time() - start))

        print('> Start to estimate Expression, Shape, and Pose!')
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        idx = 0
        SEP_list = []
        frame_list= []
        while True:
            start = time.time()

            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            faceOrNot, image = runFaceDetect(frame, net)
            cv2.imshow("Frame", frame)

            if faceOrNot == -1:
                print('no faces detected')
                # cv2.imshow("Frame", frame)
                continue

            # Fix the grey image
            if len(image.shape) < 3:
                image_r = np.reshape(image, (image.shape[0], image.shape[1], 1))
                image = np.append(image_r, image_r, axis=2)
                image = np.append(image, image_r, axis=2)

            image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
            (Shape_Texture, Expr, Pose) = sess.run([fc1ls, fc1le, pose_model.preds_unNormalized],
                                                   feed_dict={x: image})

            Pose = np.reshape(Pose, [-1])
            Shape_Texture = np.reshape(Shape_Texture, [-1])
            # Shape = Shape_Texture[0:99]
            # Shape = np.reshape(Shape, [-1])
            Expr = np.reshape(Expr, [-1])
            SEP, _ = utils.projectBackBFM_withEP(model, Shape_Texture, Expr, Pose)

            mesh_name = mesh_folder + '/' + str(idx)
            utils.write_ply_textureless(mesh_name + '_Shape_Expr_Pose.ply', SEP, faces)
            pcd = open3d.io.read_point_cloud(mesh_name + '_Shape_Expr_Pose.ply')
            open3d.visualization.draw_geometries([pcd])

            # SEP_list.append(SEP)
            # frame_list.append(frame)

            print(time.time() - start)

            key = cv2.waitKey(1) & 0xFF

            idx += 1
            if idx > 20:
                break

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # write_meshes(frame_list, SEP_list, faces)
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()


def write_meshes(frames, SEPs, faces):
    start = time.time()
    mesh_folder = './output_ply'  # The location where .ply files are saved
    if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)

    for i, SEP in enumerate(SEPs):
        frame_name = mesh_folder + '/' + str(i) + '.jpg'
        cv2.imwrite(frame_name, frames[i])
        mesh_name = mesh_folder + '/' + str(i)
        utils.write_ply_textureless(mesh_name + '_Shape_Expr_Pose.ply', SEP, faces)
    print("Writing meshes complete {}".format(time.time() - start))


def main(_):
    with tf.device('/cpu:0'):
        face_reconstruction()

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="2"
    # if FLAGS.num_gpus == 0:
    #         dev = '/cpu:0'
    # elif FLAGS.num_gpus == 1:
    #         dev = '/gpu:0'
    # else:
    #         raise ValueError('Only support 0 or 1 gpu.')

    # #print dev
    # with tf.device(dev):
    #         extract_PSE_feats()


if __name__ == '__main__':
    tf.compat.v1.app.run()
