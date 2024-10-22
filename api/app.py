# 메인 플라스크 어플리케이션
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
   return 'ㅎㅇ?? flask 서버임'

if __name__ == '__main__':  
   app.run('0.0.0.0',port=5000,debug=True)