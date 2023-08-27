from django.shortcuts import render, redirect
from django.views.decorators import gzip
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from django.http import StreamingHttpResponse
from camera.models import MyModel
import threading

# 모델 경로와 디바이스 설정
model = MyModel()
device = model.device
model.model.to(device)
model.model.eval()

# 카메라 관련 클래스
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 형식을 RGB로 변환
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()  # 프레임을 여기서 업데이트합니다
            if not self.grabbed:
                break


# 프레임 처리 및 예측 함수
def process_frame(frame, model):
    # 프레임 전처리
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb)
    # frame_tensor = transform(frame)
    if frame_tensor.shape[0] == 1:
        frame_tensor = frame_tensor.expand(3, -1, -1, -1)

    # Add a batch dimension
    frame_tensor = frame_tensor.unsqueeze(0)

    # Move the tensor to the appropriate device
    frame_tensor = frame_tensor.to(device)

    # 모델 예측
    with torch.no_grad():
        # Check the number of channels and expand if necessary
        if frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor.expand(3, -1, -1, -1)
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
        else:
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

        output = model.model(frame_tensor)


    # 예측 결과 가져오기
    _, predicted_idx = torch.max(output.data, 1)
    predicted_class = predicted_idx.item()
    print(predicted_class)

    return predicted_class

# 비디오 프레임을 가져오고 예측 결과를 반환하는 함수
def get_video_frame(cam, model):
    while True:
        frame = cam.get_frame()
        frame_np = np.frombuffer(frame, np.uint8)  # 바이트 데이터를 NumPy 배열로 변환
        frame_img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)  # NumPy 배열을 이미지로 디코딩
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        predicted_class = process_frame(frame_img_rgb, model)

        # 예측 결과 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_img_rgb, f"Predicted Class: {predicted_class}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame_img_rgb)
        frame_bytes = jpeg.tobytes()

        # 프레임 단위로 이미지 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# 프레임 및 예측 결과를 생성하는 제너레이터
# def gen(camera, model):
#     for frame in get_video_frame(camera, model):
#         # 프레임 단위로 이미지 반환
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# 웹캠 영상 스트리밍
def camera(request):
    try:
        cam = VideoCamera()  # 웹캠 호출
        return StreamingHttpResponse(get_video_frame(cam, model), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print("에러입니다:", str(e))
        pass

# new code
# def camera(request):
#     try:
#         cam = VideoCamera()  # 웹캠 호출
#         return render(request, 'camera/camera.html')
#     except Exception as e:
#         print("에러입니다:", str(e))
#         return render(request, 'camera/camera.html')
#
# def redirect_to_mic(request):
#     return redirect('/mic/mic')
#
# def get_camera_stream(cam, model):
#     while True:
#         frame = get_video_frame(cam, model)  # 프레임 가져오기
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # 이미지 프레임 반환
#
# def camera_stream(request):
#     cam = VideoCamera()  # 웹캠 호출
#     return StreamingHttpResponse(get_camera_stream(cam, model), content_type="multipart/x-mixed-replace;boundary=frame")
#
