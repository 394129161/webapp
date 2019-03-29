from flask import Blueprint, render_template, request, session, redirect, url_for
import os
from flask import jsonify, make_response
from www import dataPrecess
import json
import time
bp = Blueprint('app', __name__)
@bp.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('/home.html', id='a', title_name='welcome')

@bp.route('/tempindex', methods=['POST'])
def tempindex():
    if request.method == 'POST':
        IDdata = request.form.get('ID')
        return render_template('/home.html', id=IDdata, title_name='welcome')


@bp.route('/info')
def info():
    return render_template('/about.html', title_name='about')

@bp.route('/ajax', methods=['POST'])
def ajax():
    id = "0"
    if request.form.get("type") == "text":
        text = request.form.get('data')
        id = dataPrecess.dataPrecess("text", text)

    elif request.form.get("type") == "file":
        filename = request.form.get('data')
        id = dataPrecess.dataPrecess("file", filename)

    #return render_template('/home.html', id='b', title_name='welcome')
    return jsonify({"ID": id})

@bp.route('/loadjson', methods=['POST'])
def loadjson():
    id = request.form.get("ID")
    print(id)
    basepath = os.path.dirname(__file__)
    name = id + '.json'
    json_path = os.path.join(basepath, 'static\\json', name)
    f = open(json_path, encoding='utf-8')
    jsondata = json.load(f)
    return jsonify(jsondata)

@bp.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    basepath = os.path.dirname(__file__)
    upload_path = os.path.join(basepath, 'static\\uploads', file.filename)
    file.save(upload_path)
    response = make_response("save file success")
    return response


