from models import *
from app import db

db.create_all()
db.session.add(User(name='admin',email="admin@admin.com",password='admin'))
db.session.commit()