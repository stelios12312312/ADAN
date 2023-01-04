import os,sys,inspect
from os.path import isfile,join

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir2= os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,parentdir2) 

import adan



from adan.aiem.symbolic_modelling import findSymbolicExpression
from adan.aiem.genetics.genetic_programming import findFeaturesGP
from adan.aidc.utilities import *
from adan.aiem.genetics.evaluators import *
from adan.protocols import symbolic_regression_protocol

import flask

from functools import wraps
from flask import Flask, flash, request, session, redirect, url_for,render_template,json as fjson,send_from_directory
from werkzeug import secure_filename
#from random_words import RandomWords
from flask_sqlalchemy import SQLAlchemy

from io import StringIO
import base64


################
#### config ####
################

app = Flask(__name__)
app.config.from_object('_config')
db = SQLAlchemy(app)




def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            return 'You need to login first.'
    return wrap

def logout():
    session.pop('logged_in', None)
    flash('Goodbye!')
    return redirect(url_for('login'))
    
    
    
@app.route('/login', methods=['POST'])
def login():
    from models import User
    if request.method == 'POST':
        user = User.query.filter_by(name=request.authorization['username']).first()
        if user is not None:
            if user.name==request.authorization['username'] and user.password == request.authorization['password']:
                session['logged_in'] = True
                session['user_id'] = user.id
                session['username']=user.name
                flash('Welcome!')
                return 'login successful'

    return 'login unsuccessful'
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
           
           
def get_user_pass(auth_str):
    userpass = auth_str.copy()
    userpass = userpass.split(' ')[1]
    userpass = base64.b64decode(userpass)
    username = userpass.split(':')[0]
    password = userpass.split(':')[1]
    
    return username, password

#ADD TESTS FOR CORRECT FILE FORMAT
@app.route('/upload_data',methods=['POST'])
@login_required
def upload_data():
    from models import Dataset
    if request.method == 'POST':    
        #datafile = request.files['file']
        import joblib
        data=bytes(request.values['file'],'utf-8')
        dataname=request.values['name']  

        new_Data = Dataset(data=data,owner=session['username'],dataname=dataname)

        db.session.add(new_Data)
        db.session.commit()
        return 'dataset "{0}" addition to the DB successful'.format(dataname)
    else:
        return 'error dataset not added'

        

@app.route('/analyze',methods=['POST'])    
@login_required
def analyze():   
    from models import Dataset, User
    dataname = request.values['dataname']
    dataset = Dataset.query.filter_by(owner=session['username'],dataname=dataname).first()
    
    #df = StringIO(unicode(dataset.data))
    df = StringIO(dataset.data.decode())

    target=request.values['target']
    
    solutions = symbolic_regression_protocol(df,target,request.values['task'],time_allowed=request.values['time']) 
    
#    realizer=sentenceRealizerSymbolic()
#    realizer.interpretSymbolic(k)
#    res=realizer.realizeAll()
#    print("\n")
#    print(res)
    final_dict={}
    solution=0
    for m in solutions:
        equation = m.model
        performance = m.performance
        solution+=1
        dummy1="Explained {0}% of variance".format(str(performance*100))
        dummy2=str(equation)
        final_dict[solution]=(dummy1,dummy2)
        
    return flask.jsonify(final_dict)
    
    

    
if __name__=="__main__":
    from models import *
    app.run()