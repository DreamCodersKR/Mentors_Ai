from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS 라이브러리 추가
import pymysql
from truely import load_model, get_mentee_mentor_data, calculate_fit  # 필요한 모듈 임포트

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 접근 허용

# 모델 로드 (경로는 개인차가 있음
)
model = load_model(r'C:\Users\smhrd1\Desktop\Mentors\ai\models\KoELECTRA_base.pth')

@app.route('/recommend_mentor', methods=['POST'])
def recommend_mentor_endpoint():
    data = request.json
    mentee_email = data.get('mentee_email')

    if not mentee_email:
        return jsonify({'error': '멘티 이메일이 필요합니다.'}), 400

    # 멘토 추천
    mentee, mentors = get_mentee_mentor_data(mentee_email)
    if not mentee:
        return jsonify({'error': '해당 멘티를 찾을 수 없습니다.'}), 404

    if not mentors:
        return jsonify({'error': '해당 멘티와 일치하는 멘토를 찾을 수 없습니다.'}), 404

    fit_results = calculate_fit(mentee, mentors, model)
    best_mentor = fit_results[0] if fit_results else None

    if best_mentor:
        # DB에 매칭 결과 저장
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='1234',
            db='mentors',
            charset='utf8mb4'
        )
        try:
            with conn.cursor() as cursor:
                sql_insert_match = """
                    INSERT INTO tb_match_data (bios_idx, match_score, mentor_email, mentee_email)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql_insert_match, (mentee['bios_idx'], best_mentor[2], best_mentor[1], mentee['user_email']))
                conn.commit()
                print('매칭 결과가 성공적으로 저장되었습니다.')
        except Exception as e:
            conn.rollback()
            return jsonify({'error': f"매칭 결과 저장 중 오류 발생: {e}"}), 500
        finally:
            conn.close()

        # return jsonify({
        #     'mentor_name': best_mentor[0],
        #     'mentor_email': best_mentor[1],
        #     'score': best_mentor[2]
        # }), 200
    
        return jsonify({
            'mentor_name': best_mentor[0],
            'mentor_email': best_mentor[1],
            'score': float(best_mentor[2])  # float32를 float으로 변환
        }), 200

    else:
        return jsonify({'error': '매칭된 멘토가 없습니다.'}), 404

if __name__ == '__main__':
    app.run(port=5000)
