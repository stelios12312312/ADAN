from app import db


class User(db.Model):

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)
    password = db.Column(db.String, nullable=False)

    def __init__(self, name=None, email=None, password=None):
        self.name = name
        self.email = email
        self.password = password

    def __repr__(self):
        return '<User {0}>'.format(self.name)
        
class Dataset(db.Model):

    __tablename__ = 'datasets'
    __table_args__ = (db.UniqueConstraint("owner", "dataname"),)
    
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.LargeBinary, unique=False, nullable=False)
    owner = db.Column(db.String, db.ForeignKey("users.name"), unique=False, nullable=False)
    dataname = db.Column(db.String, unique=False, nullable=False)

    def __init__(self, owner=None,data=None,dataname=None):
        self.data = data
        self.owner=owner
        self.dataname=dataname

    def __repr__(self):
        return '<Dataset {0}>'.format(self.id)