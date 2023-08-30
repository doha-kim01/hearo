from .views import *
from django.urls import path
from . import views

app_name = "cam"

urlpatterns = [
    path('camera', camera, name='camera'),
    # path('mic/mic', views.redirect_to_mic, name='redirect_to_mic'),
    # path('camera/stream', views.camera_stream, name='camera_stream'),
    # 카메라 스트리밍을 위한 URL 패턴 추가
]