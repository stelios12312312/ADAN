import pandas as pd
import os


# grab the folder where this script lives
basedir = os.path.abspath(os.path.dirname(__file__))

DATABASE = 'main.db'
CSRF_ENABLED = True
SECRET_KEY = 'my_precious'

# define the full path for the database
DATABASE_PATH = os.path.join(basedir, DATABASE)

# the database uri
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + DATABASE_PATH
SQLALCHEMY_TRACK_MODIFICATIONS = None

ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
ANALYSIS_FOLDER = os.path.join(basedir, 'analysis')
DATA=pd.DataFrame()