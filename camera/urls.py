from .views import *
from django.urls import path

app_name = "cam"

urlpatterns = [
    path('camera', camera, name='camera'),
]