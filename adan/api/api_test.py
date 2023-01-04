import os
import unittest
import tempfile
#import app
from werkzeug.datastructures import Headers
import base64
from werkzeug.test import Client
from random_words import RandomWords
import pandas as pd
import io
import requests


class FlaskrTestCase(unittest.TestCase):
    
#    def setUp(self):
#        self.app = app.app.test_client()
    
#    def login(self, username, password):
#        head = Headers()
#        head.add('authorization', 'Basic ' + username+":"+password)
#        rv = Client.post(self.app, path='/login',headers=head)
#        return rv
#        #paok=requests.post('http://127.0.0.1:5000/login', auth=(username, password))
    
#    def logout(self):
#        return self.app.get('/logout', follow_redirects=True)
#        
#    def test_login(self):
#        sms = self.login('admin','admin')
#        assert(sms.data.find('successful')>-1)
        
#    def test_upload(self):
#        
#        self.login('admin','admin')
#        rw = RandomWords()
#        data={}
#        data['name']=rw.random_word()
#        data['file']=pd.read_csv('auto_mpg.csv').to_csv()
#        #self.upload_data2()
#        #self.upload_data('paok','auto_mpg')
#        sms = self.app.post('/upload_data', data=data,follow_redirects=True)
#        assert(sms.data.find('successful')>-1)
        
#    def test_upload_dummy(self):
#        sess = requests.Session()
#        k = sess.post('http://127.0.0.1:5000/login', auth=('admin', 'admin'))
#        rw = RandomWords()
#        data={}
#        data['name']=rw.random_word()
#        data['file']=pd.read_csv('auto_mpg.csv').to_csv()
#        #self.upload_data2()
#        #self.upload_data('paok','auto_mpg')
#        sms = sess.post('http://127.0.0.1:5000/upload_data', data=data)
#        assert(sms.data.find('successful')>-1)
        
        
        
    def test_analyze_dummy(self):
        sess = requests.Session()
        k = sess.post('http://127.0.0.1:5000/login', auth=('admin', 'admin'))
        rw = RandomWords()
        data={}
        dataname = rw.random_word()
        data['name']=dataname
        data['file']=pd.read_csv('auto_mpg.csv').to_csv()
        #self.upload_data2()
        #self.upload_data('paok','auto_mpg')
        sms = sess.post('http://127.0.0.1:5000/upload_data', data=data)
        
        
        data={}
        data['target']='V1'
        data['dataname']=dataname
        data['task']='regression'
        data['time']=5
        sms = sess.post('http://127.0.0.1:5000/analyze',data=data)
        response = sms.content.decode()
        assert(response.find('variance')>-1)
        
#    def test_analyze(self):
#        self.login('admin','admin')
#        rw = RandomWords()
#        dataname=rw.random_word()
#        data={}
#        data['name']=dataname
#        data['file']=pd.read_csv('auto_mpg.csv').to_csv()
#        
#
#        self.app.post('/upload_data', data=data,follow_redirects=True)
#        
#        data={}
#        data['target']='V1'
#        data['dataname']=dataname
#        data['task']='regression'
#        data['time']=5
#        sms = self.app.get('/analyze',data=data,follow_redirects=True)
#        assert(sms.data.find('variance')>-1)
                      
        
    

if __name__ == '__main__':
    unittest.main()