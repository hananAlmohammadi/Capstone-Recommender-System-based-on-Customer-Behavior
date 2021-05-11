from inference import get_id, brand_list
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for
app = Flask(__name__)

import os

@app.route('/', methods=['GET', 'POST'])
def recommendation():
    # Write the GET Method to get the index file
    # Write the POST Method to post the results file
    if request.method =='POST':
        user = request.form['nm']
        print(request.form)
        if 'nm' not in request.form:
            print('File Not Uploaded')
        # Read input 
        nm = request.form['nm']
        # # Get recommendation
        recomend,price1,brand1= get_id(nm)
        # import PIL
        
        # from PIL import Image
        relpath='/static/' +brand1+'.jpg'
        image_names = os.listdir('C:/Users/hano0/Desktop/DSI8/capstone/DataArmors/static')
        return render_template('recomender.html',p_id=recomend,price= price1,brand=brand1 ) 
    if request.method == 'GET':
        user = request.args.get('nm')
        list1=brand_list()
        return render_template('page.html',list2=list1)
if __name__ == '__main__':
    app.run(debug=True)