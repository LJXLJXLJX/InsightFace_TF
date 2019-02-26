import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.platform import gfile
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from time import time


def graphDef():
    x = tf.placeholder(name='img_inputs', shape=[None, 112, 112, 3], dtype=tf.float32)
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    w_init = w_init_method
    net = get_resnet(x, 50, type='ir', w_init=w_init_method, trainable=False)
    with tf.Session() as sess:
        graph_def = sess.graph.as_graph_def()
        with tf.gfile.FastGFile('./model/test.pb', 'wb') as f:
            f.write(graph_def.SerializeToString())


def freezeGraph():
    graphDef()
    freeze_graph(input_graph='./model/test.pb',  # =some_graph_def.pb
                 input_saver="",
                 input_checkpoint='./ckpt/InsightFace_iter_best_710000.ckpt',
                 checkpoint_version=2,
                 output_graph='./model/out.pb',
                 input_binary=True,
                 restore_op_name="save/restore_all",
                 filename_tensor_name="save/Const:0",
                 initializer_nodes="",
                 variable_names_whitelist="",
                 variable_names_blacklist="",
                 input_meta_graph="",
                 saved_model_tags='serve',
                 clear_devices=True,
                 output_node_names='resnet_v1_50/E_BN2/Identity',
                 )


def lookUpPb(pb_path):
    with tf.Session() as sess:
        with open(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print(graph_def)


def predictWithPb(img_path, pb_path='./model/out.pb'):
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    start = time()
    with tf.Session() as session:
        img = cv2.imdecode(np.fromfile(img_path, np.uint8), 1)
        img = cv2.resize(img, (112, 112))

        img = img - 127.5
        img = img / 128.0

        img = img.reshape((1, 112, 112, 3))
        prediction_tensor = session.graph.get_tensor_by_name('resnet_v1_50/E_BN2/Identity:0')
        output = session.run(prediction_tensor, {'img_inputs:0': img})[0]
    print(time() - start)
    return output



def visualPb(pb_path):
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    graph_def.ParseFromString(gfile.FastGFile(pb_path, 'rb').read())
    tf.import_graph_def(graph_def, name='graph')
    summaryWriter = tf.summary.FileWriter('./log/', graph)


def dnnTest(img_path,model_path):
    img= cv2.imdecode(np.fromfile(img_path, np.uint8), 1)
    img = cv2.resize(img, (112, 112))
    img = img - 127.5
    img = img / 128.0
    img=img.reshape(1,3,112,112)
    net=cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setInput(img)
    out = net.forward()
    return out




if __name__ == '__main__':
    # feat1 = predictWithPb('./test_img/2/face_0.jpg')
    # feat2 = predictWithPb('./test_img/1.jpg')
    #
    # feat1 = feat1.reshape(1, -1)
    # feat2 = feat2.reshape(1, -1)
    #
    # feat1 = normalize(feat1)
    # feat2 = normalize(feat2)
    #
    # dist = np.subtract(feat1, feat2)
    # dist = np.sqrt(norm(dist))
    # print('sim:',(2 - dist)/2)

    out=dnnTest('./test_img/1.jpg','./model/out.pb')
    print(out)
    # visualPb('./model/out.pb')