from flask import Flask
#创建app应用,__name__是python预定义变量，被设置为使用本模块.
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import *

nav=Nav()
nav.register_element('top', Navbar(u'文本分类',
                                    View(u'分类', 'home'),
                                    View(u'关于', 'about'),
))

app = Flask(__name__)
Bootstrap(app)
nav.init_app(app)
#如果你使用的IDE，在routes这里会报错，因为我们还没有创建呀，为了一会不要再回来写一遍，因此我先写上了
from www import routes
