from flask import Blueprint, render_template, request, session, redirect, url_for
import os
from flask import jsonify
bp = Blueprint('app', __name__)
json = []
@bp.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':

        if request.form.get('texts'):
            texts = request.form.get('texts')
            #json.append({"item": '5', "type": '6'})
            #return url_for('app.ajax')
        elif request.files['file']:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            upload_path = os.path.join(basepath, 'static\\uploads', f.filename)
            f.save(upload_path)

            #json.append({"item": '7', "type": '8'})
        ajax()
        json.clear()
        return redirect(url_for('app.index'))


    if request.method == 'GET':
        return render_template('/home.html', title_name='welcome')



@bp.route('/info')
def info():
    return render_template('/about.html', title_name='about')

@bp.route('/ajax')
def ajax():
    #json = [{"item": '1', "type": '2'}, {"item": '3', "type": '4'}]
    return jsonify(json)

