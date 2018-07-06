from flask import Flask, render_template, request, url_for
from scipy.misc import imread, imresize
import numpy as np
import re
import os
from load import init
import base64

OUTPUT_FILE = 'output.png'

app = Flask(__name__)

global model
model = init()


@app.route('/')
def index():
    if not request.script_root:
        request.script_root = url_for('index')
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img_data = request.get_data()
    convert_img(img_data)
    x = imread(OUTPUT_FILE, mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)

    out = model.predict(x)
    res = np.array_str(np.argmax(out, axis=1))
    return res


def convert_img(img_data):
    img_str = re.search(rb'base64,(.*)', img_data).group(1)
    with open(OUTPUT_FILE, 'wb') as output:
        output.write(base64.b64decode(img_str))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
