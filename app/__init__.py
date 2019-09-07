from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap

app = Flask(__name__)
boostrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'you-will-never-guess'
from app import routes


