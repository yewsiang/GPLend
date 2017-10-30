from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/borrowers')
def borrowers():
    return render_template('borrowers.html')

@app.route('/managers')
def managers():
    return render_template('index.html')