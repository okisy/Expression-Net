import sys
import numpy as np
import tensorflow as tf
import os.path
import ST_model_nonTrainable_AlexNetOnFaces as Pose_model

from realtime_estimating import realtime_estimating

sys.path.append('./kaffe')
sys.path.append('./ResNet')
from ThreeDMM_shape import ResNet_101 as resnet101_shape
from ThreeDMM_expr import ResNet_101 as resnet101_expr

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 227, 'Image side length.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size')

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


def extract_PSE_feats():
    # placeholders for the batches
    x = tf.compat.v1.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])

    ###################
    # Face Pose-Net
    ###################
    net_data = np.load("./fpn_new_model/PAM_frontal_ALexNet_py3.npy").item()
    pose_labels = np.zeros([FLAGS.batch_size, 6])
    x1 = tf.compat.v1.image.resize_bilinear(x, tf.constant([227, 227], dtype=tf.int32))

    # Image normalization
    x1 = x1 / 255.  # from [0,255] to [0,1]
    # subtract training mean
    mean = tf.reshape(train_mean_vec, [1, 1, 1, 3])
    mean = tf.cast(mean, 'float32')
    x1 = x1 - mean

    pose_model = Pose_model.Pose_Estimation(x1, pose_labels, 'valid', 0, 1, 1, 0.01, net_data, FLAGS.batch_size,
                                            mean_labels, std_labels)
    pose_model._build_graph()
    del net_data
    print('loading coefficients regression model.')
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
    saver_ini_expr_net = tf.compat.v1.train.Saver(
        var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))
    
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)

        # load our expression net model
        load_path = "./Expression_Model/ini_exprNet_model.ckpt"
        saver_ini_expr_net.restore(sess, load_path)

        print ('> Start to estimate Expression, Shape, and Pose!')

        realtime_estimating(sess, FLAGS, fc1le, pose_model, x)


def main(_):
    with tf.device('/cpu:0'):
        extract_PSE_feats()
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
