from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap
from www import app

@app.route('/')
def home():
    return render_template('home.html', title_name='welcome')

@app.route('/about')
def about():
    return render_template('base.html', title_name='about')
