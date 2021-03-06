# !/usr/bin/env python
import os
from www import create_app
#from www.models import User, Role
from flask_script import Manager, Shell
#from flask.ext.migrate import Migrate, MigrateCommand

app = create_app(os.getenv('FLASK_CONFIG') or 'default')
manager = Manager(app)
#migrate = Migrate(app, db)


def make_shell_context():
    return dict(app=app)


manager.add_command("shell", Shell(make_context=make_shell_context))
#manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
