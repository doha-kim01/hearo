# 청각장애인을 위한 의료 기관에서의 쌍방향 소통 웹페이지 개발
* 이화여자대학교 도전학기 15기 사회문제해결형 선정  
* ICT멘토링 이브와 프로젝트 선정

## Web tutorial

`in MacOS Ventura 13.3.1`

make a folder to manage
```
mkdir example_folder
cd example_folder # 해당 폴더로 이동
```
가상환경 생성 후 활성화
```
python -m venv 가상환경명
cd 가상환경명
source ./bin/activate
```
**아래 과정부터는 모두 가상환경이 활성화 된 이후 진행합니다.**  

Install Django & pip upgrade
```
pip install django
python -m pip install --upgrade pip v # 필수는 아님
```
Project clone
```
git clone https://github.com/HearoEwha/hearo.git
```
Move to project folder 
```
cd hearo
```
run server
```python
python manage.py runserver
```
## Troubleshooting

**pyaudio 오류**

Install portaudio
```
brew install portaudio
```
Link portaudio
```
brew link portaudio
```
Copy the path where portaudio was installed (use it in the next step)
```
brew --prefix portaudio #여기서 나온 경로 복붙해서 보관
```
Create .pydistutils.cfg in your home directory
```
sudo nano $HOME/.pydistutils.cfg
```
then paste the following
```
[build_ext]
include_dirs=<PATH FROM STEP 3>/include/
library_dirs=<PATH FROM STEP 3>/lib/
```
Install pyaudio
```
pip install pyaudio
or
pip3 install pyaudio
```

## 출처

https://stackoverflow.com/questions/73268630/error-could-not-build-wheels-for-pyaudio-which-is-required-to-install-pyprojec

https://curryyou.tistory.com/140
