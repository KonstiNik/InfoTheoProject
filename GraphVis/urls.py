from django.conf.urls import include, url
from django.contrib import admin
from django.urls import include, path

from . import views

urlpatterns = [
    path('', views.index2, name='index2'),
    path('addVertex', views.index, name='index'),
    path('removeVertex', views.removeVertex, name='removeVertex'),
    path('test',views.init, name='init'),
    path('addEdge', views.addEdge, name='addedge'),
    path('removeEdge', views.removeEdge, name='removeEdge'),
    path('drawMethod', views.drawMethod, name='drawMethod'),
    path('loadSide', views.loadSide, name='loadSide'),
    path('compute', views.compute, name='compute'),
]

#urlpatterns = ['',
 #   url(r'^GraphVis/', 'GraphVis.views.index'),
  #  url(r'^GraphVis/send/', 'GraphVis.views.search'),
#]