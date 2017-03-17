from PIL import Image

import numpy as np
import requests
import tensorflow as tf
from StringIO import StringIO
from scipy import misc
from src import transform, utils


def feed_forward(image_data, checkpoint_dir, device_t='/gpu:0'):
    img_shape = image_data.shape
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), tf.Session(
            config=soft_config) as sess:
        batch_shape = (1,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        model = transform.net(img_placeholder)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_dir)
        output = sess.run(model, feed_dict={
            img_placeholder: np.array([image_data])})
        output = output.reshape(*output.shape[1:])
        return output


def apply_filter(image_or_url, filter_name, intensity=0.75):
    if isinstance(image_or_url, str):
        response = requests.get(image_or_url)
        img = np.asarray(Image.open(StringIO(response.content)))
    else:
        img = np.asarray(img_or_url)
    aimg = utils.border_img(img)
    output = feed_forward(aimg, "%s/fns.ckpt" % filter_name)
    output = np.clip(output, 0, 255)
    output = utils.crop_img(output)
    output = misc.imresize(output, img.shape)
    output = img * (1 - intensity) + output * (intensity)

    return Image.fromarray(output.astype(np.uint8))


if __name__ == '__main__':
    apply_filter(
        'http://2.bp.blogspot.com/-lWBJhJTyEic/UZKe3zDVQiI/AAAAAAAAI_A/vBcbVTZdLm4/s1600/berlu.jpg',
        'rain_princess').save('out/bello2.jpg')

