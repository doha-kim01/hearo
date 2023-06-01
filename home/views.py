
from django.shortcuts import render

# Create your views here.
def select(request):
    return render(request, 'home/select.html')


def cam(request):
    return render(request, 'home/camera.html')
