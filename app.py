from flask import Flask
from flask import request
from flask import render_template
import tensorflow as tf
import forward
from PIL import Image
import numpy as np

app = Flask(__name__)


# ===============================================================
# basic model
def basic_m(xs):
    X = tf.placeholder(tf.float32, [None, None, None, 3])
    with tf.name_scope('generator'), tf.variable_scope('generator'):
        Y = forward.forward(X, 1, False)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    xs = (xs / 128) - 1
    img = sess.run(Y, feed_dict={X: xs})
    img = (img + 1) / 2
    img *= 256
    img = img.astype(np.uint8)
    # Image.fromarray(img[0]).save('test.png')
    return img


# ==============================================================
# one piece model
def one_piece(xs):
    xs = (xs / 255. - 0.5) * 2
    one_sess = tf.Session()

    one_sess.run(tf.global_variables_initializer())

    one_saver = tf.train.import_meta_graph('./one512_10000-10000.meta')
    one_saver.restore(one_sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    g = graph.get_tensor_by_name('generator/g:0')
    X = graph.get_tensor_by_name('X:0')

    img = one_sess.run(g, feed_dict={X: xs})
    img = img / 2 + 0.5
    img *= 255
    img = img.astype(np.uint8)
    return img


# todo get result image from /static/upload/img to /static/result/img by dataset model
def get_result(dataset, name):
    xs = np.array(Image.open('static/upload/' + name).resize((512, 512)).convert('RGB'))
    xs = np.expand_dims(xs, 0)
    # basic
    if dataset == 1:
        img = basic_m(xs)
    elif dataset == 2:
        img = one_piece(xs)
    Image.fromarray(img[0]).save('static/result/' + name)


@app.route('/')
@app.route('/basic')
def basic():
    return render_template('index.html')


@app.route('/upload/image', methods=['POST'])
def upload_image():
    dataset = request.args.get('dataset')
    print(dataset)
    f = request.files['file']
    filename = 'static/upload/' + f.filename
    f.save(filename)

    get_result(int(dataset), f.filename)
    print(dataset, f.filename)
    return f.filename


if __name__ == '__main__':
    app.run()
