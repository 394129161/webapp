from flask import Blueprint, render_template, request, session, redirect, url_for
import os
from flask import jsonify
import json
import time
bp = Blueprint('app', __name__)
@bp.route('/')
def index():
    if len(request.files) != 0:
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static\\uploads', f.filename)
        f.save(upload_path)
    return render_template('/home_try.html', title_name='welcome')



@bp.route('/info')
def info():

    return render_template('/about.html', title_name='about')

@bp.route('/ajaxone', methods=['POST'])
def ajaxone():
    str = request.form['str']
    rtnData = {"item": str, "type": '6'}
    return jsonify(rtnData)


@bp.route('/ajaxfile', methods=['POST'])
def ajaxfile():
    name = request.form['name']
    rtnData = {"item": name, "type": '6'}
    return json.dumps(rtnData)


