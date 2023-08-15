from django.shortcuts import render
from django.views.decorators import gzip
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from django.http import StreamingHttpResponse
from camera.models import MyModel
import threading
from PIL import ImageFont, ImageDraw, Image
import torch.nn.functional as F

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

#-*- coding: utf-8 -*- 

# 프레임 처리 및 예측 함수
def process_frame(frame, model):
    # 프레임 전처리
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
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
        probabilities = F.softmax(output, dim=1)
        
    class_mapping = {
        0: "가슴",
        1: "귀",
        2: "너무 아파요",
        3: "머리",
        4: "목",
        5: "무릎",
        6: "발",
        7: "발가락",
        8: "발목",
        9: "배",
        10: "손가락",
        11: "손목",
        12: "어깨",
        13: "팔꿈치",
        14: "허리"
        # 추가적인 클래스와 번호를 매핑하면 됩니다.
    }


    # 예측 결과 가져오기
    _, predicted_idx = torch.max(output.data, 1)
    predicted_class_num = predicted_idx.item()
    #print(predicted_class)
    predicted_class_name = class_mapping.get(predicted_class_num, "알 수 없음")
    predicted_probability = probabilities[0][predicted_idx].item()
    
    print(predicted_class_name)
    print(f"Class Probability: {predicted_probability:.2f}")
    
    return predicted_class_name, predicted_probability

# 비디오 프레임을 가져오고 예측 결과를 반환하는 함수
def get_video_frame(cam, model):
    while True:
        frame = cam.get_frame()
        frame_np = np.frombuffer(frame, np.uint8)  # 바이트 데이터를 NumPy 배열로 변환
        frame_img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)  # NumPy 배열을 이미지로 디코딩
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        predicted_class, predicted_probability = process_frame(frame_img_rgb, model)
        print(predicted_class, predicted_probability)
        
        pil_image = Image.fromarray(frame_img_rgb)

        # 예측 결과 표시
        #font = cv2.FONT_HERSHEY_SIMPLEX
        font = ImageFont.truetype('/Users/song-yeojin/hearoweb/hearo/static/fonts/NanumGothic.ttf',40)
        draw = ImageDraw.Draw(pil_image)
        text = f"Predicted Class: {predicted_class}"
        text_prob = f"Probability: {predicted_probability:.2f}"
        text_position = (10, 30)
        text_position_prob = (10, 80) #확률 위치
        text_color = (0, 255, 0)  # Green color (RGB format)
        draw.text(text_position, text, font=font, fill=text_color) # 클래스
        draw.text(text_position_prob, text_prob, font=font, fill=text_color) #확률
        
        
        frame_img_rgb = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        #cv2.putText(frame_img_rgb, f"Predicted Class: {predicted_class}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
