import os
from flask import Flask,request, url_for, redirect, render_template, flash
from werkzeug.utils import secure_filename
import pickle
import pandas as pd
import numpy as np

model=pickle.load(open('model.pkl','rb'))

UPLOAD_FOLDER = './static/files'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if allowed_file(file.filename)==False:
            flash('Not supported filetype')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_location = os.path.join(
                app.config['UPLOAD_FOLDER'], filename)
            # file.save(file_location)
            input = pd.read_csv(file)
            final = input.iloc[:,1:-1].values
            print(final)
            print(model.predict_proba(final))
            # return redirect(url_for('upload_predict',
            #                        filename=filename))
            return render_template("index.html", prediction=1)
    return render_template("index.html", prediction = 0)

if __name__ == '__main__':
    app.run(debug=True)
