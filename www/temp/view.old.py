from flask import Blueprint, render_template, request, session, redirect, url_for
import os
from flask import jsonify
import time
bp = Blueprint('app', __name__)
json = []
@bp.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':

        if request.form.get('texts'):
            texts = request.form.get('texts')
            json = ({"item": texts, "type": '6'})
            return render_template('/home_try.html', title_name='welcome', data=jsonify(json))
        elif len(request.files) != 0:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_path = os.path.join(basepath, 'static\\uploads', f.filename)
            f.save(upload_path)

            #json.append({"item": '7', "type": '8'})
        #return redirect(url_for('app.index'))
        return render_template('/home_try.html', title_name='welcome')


    if request.method == 'GET':
        return render_template('/home_try.html', title_name='welcome')



@bp.route('/info')
def info():
    return render_template('/about.html', title_name='about')

@bp.route('/ajax')
def ajax():
    #json = [{"item": '1', "type": '2'}, {"item": '3', "type": '4'}]
    return jsonify(json)

