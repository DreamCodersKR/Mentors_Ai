# 메인 플라스크 어플리케이션
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/')
def home():
   return 'ㅎㅇ?? flask 서버임'

@app.route('/receiveTestData', methods=['POST'])
def receive_data():
    data = request.json  # Java에서 보낸 JSON 데이터를 받음
    print(f"Java로부터 데이터 받음: {data}")
    
    # 간단한 응답
    response = {'status': 'success!!', 'received': data}
    return jsonify(response)

if __name__ == '__main__':  
   app.run('0.0.0.0',port=5000,debug=True)