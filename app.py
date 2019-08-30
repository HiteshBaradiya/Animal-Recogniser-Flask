from tensorflow.keras.models import load_model
import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)
APP_ROOT = 'Images/'

@app.route("/")
def index():
    return render_template("imageupload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = APP_ROOT
    if not os.path.isdir(target):
        os.mkdir(target)
#    for file in request.files.getlist("file"):
    file = request.files['file']
    filename = file.filename
    destination = APP_ROOT + filename
    file.save(destination)
    im = image.load_img(destination, target_size=(300,300))
    c = np.array(im)
    c = np.expand_dims(c, axis=0)
    model = load_model('model.h5')
    x = model.predict(c)
    d = {'result' : int(x[0][0])}
    return jsonify(d)
    
        
if __name__ == "__main__":
    app.run(port=4555,debug=True)
   
