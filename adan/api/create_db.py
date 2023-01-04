from datetime import date
from models import User


# create the database and the db table
db.create_all()

# insert data
# db.session.add(Task("Finish this tutorial", date(2015, 3, 13), 10, 1))
db.session.add(User("admin", 'stylianos.kampakis@gmail.com', 'admin'))

# commit the changes
db.session.commit()