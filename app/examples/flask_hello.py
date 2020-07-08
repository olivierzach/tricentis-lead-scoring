from flask import Flask, make_response

app = Flask(__name__)


@app.route('/')
def index():
    return '<h1>hello world!</h1>'

# to run this app - issue the following in a terminal
# % export FLASK_APP=flask_hello.py
# % flask run


@app.route('/user/<name>')
def user(name):
    return f"<h1>hello {name}!</h1>"

# see the url map
# from another python script...
# from hello_flask import app
# app.url_map
# Map([<Rule '/' (HEAD, OPTIONS, GET) -> index>,
#  <Rule '/static/<filename>' (HEAD, OPTIONS, GET) -> static>,
#  <Rule '/user/<name>' (HEAD, OPTIONS, GET) -> user>])

