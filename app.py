

import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model(r"D:/KEC/Student Project and Papers/Final Year students Project Paper - 2022 batch IT - Hari, Maheshwaran and Dinesh/Solar_Panel_Defect_Identification Running as on 24-03-2022/solar_panel_defect_Identification/Solar_Panel_Defect_Identification.h5")
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (128,128)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis=0)
        print(x)
        preds = model.predict(x)
        pred=np.argmax(preds,axis=1)
        print("prediction",pred)
        index = ['bad_condition','good_condition']
        text = " Solar panel is in " + str(index[pred[0]])
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)