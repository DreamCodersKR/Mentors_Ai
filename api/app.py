# 메인 플라스크 어플리케이션
from flask import Flask, jsonify, request
import urllib.parse
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

# GET 요청 처리
@app.route('/receiveTestGetData', methods=['GET'])
def receive_get_data():
    # GET 요청으로 URL에서 'message' 파라미터를 받음
    message = request.args.get('message')
    # 인코딩된 데이터를 디코딩할 필요가 있음
    decoded_message = urllib.parse.unquote(message) # URL 인코딩 해제
    print(f"Java 서버로부터 데이터 받음: {decoded_message}")
    
    # 응답 반환
    response = {'status': '성공함!', 'received_data': decoded_message}
    return jsonify(response)

if __name__ == '__main__':  
   app.run('0.0.0.0',port=5000,debug=True)