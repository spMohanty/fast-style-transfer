import tensorflow as tf
import transform
import redis
import numpy as np
from PIL import Image
import io

import tensorflow as tf
from tensorflow.python.framework import ops

import utils

import time

from flask import Flask, render_template, Response
import cv2

HOST="localhost"

class TimeTravellingMicroscopeTester:

    def __init__(self, content_image, model_path):
        # session

        # soft_config = tf.ConfigProto(allow_soft_placement=True)
        # soft_config.gpu_options.allow_growth = True # to deal with large image

        

        # input images
        self.x0 = content_image

        self.image_shape = (480, 640, 3)

        # input model
        self.model_path = model_path

        # image transform network
        self.transform = transform.Transform()

        # build graph for style transfer
        self.lock = False
        
        self.load_model()

    def reset_session(self):
        ops.reset_default_graph()
        sess = tf.InteractiveSession()   
        return sess     

    def _build_graph(self):

        # graph input
        self.x = tf.placeholder(tf.float32, shape=self.x0.shape, name='input')
        self.xi = tf.expand_dims(self.x, 0) # add one dim for batch

        # result image from transform-net
        self.y_hat = self.transform.net(self.xi/255.0)
        self.y_hat = tf.squeeze(self.y_hat) # remove one dim for batch
        self.y_hat = tf.clip_by_value(self.y_hat, 0., 255.)

    def load_model(self, model_path=False):
        # initialize parameters
        if self.lock:
            print("Model Load in progress....ignoring current request..")
            return
        
        self.lock = True
        self.sess = self.reset_session()
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())
        

        # load pre-trained model
        saver = tf.train.Saver()
        if not model_path:
            model_path = self.model_path
        
        print("Loading model : ", model_path)
        saver.restore(self.sess, model_path)
        self.lock = False

    def transfer_style(self, content_image):
        # get transformed image
        if self.lock:
            return content_image

        output = self.sess.run(self.y_hat, feed_dict={self.x: content_image})
        return output



####
## Main
if __name__ == "__main__":
    redis_cli = redis.Redis(HOST, 6379)

    ## Load Model
    

    MODELS = ['models/la_muse.ckpt', 'models/rain_princess.ckpt', 'models/shipwreck.ckpt', 'models/the_scream.ckpt', 'models/udnie.ckpt', 'models/wave.ckpt']
    CURRENT_STYLE = 1
    style_path = MODELS[CURRENT_STYLE]

    transformer = False

    def get_next_frame():
        img_bytes = redis_cli.get("I")
        decoded = np.array(Image.open(io.BytesIO(img_bytes)), dtype=np.float32)
        return decoded

    content_image = get_next_frame()

    transformer = TimeTravellingMicroscopeTester(
                    model_path=style_path,
                    content_image=content_image,
                    )
    app = Flask(__name__)
    @app.route('/')
    def index():
        """home page."""
        return render_template('index.html')

    @app.route('/next_style')
    def next_style():
        """
        Move to next style
        """

        global CURRENT_STYLE
        CURRENT_STYLE += 1
        style_path = MODELS[CURRENT_STYLE % len(MODELS)]
        transformer.load_model(style_path)

        return "Success\n"

    def gen():
        """Video streaming generator function."""
        while True:
            frame = get_next_frame()

            # global CURRENT_STYLE
            # CURRENT_STYLE += 1
            # style_path = MODELS[CURRENT_STYLE % len(MODELS)]
            # transformer.load_model(style_path)

            output_image = transformer.transfer_style(frame)
            utils.save_image(output_image, "result.jpeg")

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + open("result.jpeg", "rb").read() + b'\r\n')

    @app.route('/video_feed')
    def video_feed():
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


    def source_gen():
        """Video streaming generator function."""
        while True:
            frame = get_next_frame()
            utils.save_image(frame, "frame.jpeg")

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + open("frame.jpeg", "rb").read() + b'\r\n')
    
    @app.route('/source_video_feed')
    def source_video_feed():
        return Response(source_gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


    app.run(host='0.0.0.0', debug=True)

    # start_time = time.time()
    # output_image = transformer.transfer_style(content_image)
    # end_time = time.time()
    # print(end_time-start_time)          
    # utils.save_image(output_image, "result.jpeg")








