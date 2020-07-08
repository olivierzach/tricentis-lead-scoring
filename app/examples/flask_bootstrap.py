from flask import Flask, render_template
from flask_bootstrap import Bootstrap


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
    return '<h1>hello world!</h1>'


@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)


if __name__ == '__flask_bootstrap__':
    app.run()
