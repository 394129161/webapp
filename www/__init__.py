from flask import Flask
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import *
from config import config

nav = Nav()
nav.register_element('top', Navbar(u'文本分类',
                                   View(u'分类', 'app.index'),
                                   View(u'关于', 'app.info'),
                                   )
                     )
bootstrap = Bootstrap()


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name]) # 可以直接把对象里面的配置数据转换到app.config里面
    config[config_name].init_app(app)

    bootstrap.init_app(app)
    nav.init_app(app)
    #mail.init_app(app)
    #moment.init_app(app)
    #db.init_app(app)
    # 路由和其他处理程序定义
    from www import view
    app.register_blueprint(view.bp)
    app.add_url_rule('/', endpoint='index')
    return app


from www import view
