import re
import torch
import torch.nn as nn
from nltk.corpus import stopwords as nltk_stopwords
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ElectraPreTrainedModel, ElectraModel, AutoTokenizer
import pymysql

# GPU 사용 가능 여부에 따라 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 한국어 불용어 목록 정의 (필요 시 실제 불용어 목록을 추가하세요)
korean_stopwords = []

# 영어 불용어 목록 정의
english_stopwords = nltk_stopwords.words('english')

# 형태소 분석기 인스턴스 생성
okt = Okt()

# KoBERT 모델의 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# 맞춤형 모델 클래스 정의
class CustomElectraForSequenceClassification(ElectraPreTrainedModel):
    # 경고 메시지를 해결하기 위해 학습이 필요할 수 있습니다. 모델 학습을 진행하세요.
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 분류할 레이블의 수 설정
        self.electra = ElectraModel(config)  # 기본 Electra 모델 불러오기
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 분류를 위한 출력 레이어 정의
        self.init_weights()  # 가중치 초기화

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_hidden_states=True,
        return_dict=None,
    ):
        # return_dict 설정
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Electra 모델을 사용하여 출력 얻기
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state  # 시퀀스 출력 (batch_size, seq_length, hidden_size)
        pooled_output = sequence_output[:, 0, :]  # [CLS] 토큰의 임베딩 추출

        logits = self.classifier(pooled_output)  # 분류 결과 계산

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        # 결과를 딕셔너리 형태로 반환
        return {'logits': logits, 'hidden_states': outputs.hidden_states}

# 모델 로드 함수 정의
def load_model(model_path='../models/KoELECTRA_base.pth'):
    model = CustomElectraForSequenceClassification.from_pretrained(
        'monologg/koelectra-base-v3-discriminator', num_labels=1, ignore_mismatched_sizes=True
    )

    # 모델 가중치 로드 시 device 지정
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 모델을 device로 이동
    model.to(device)
    model.eval()
    return model

# 전처리 함수 정의
def preprocess_text(text):
    # 전처리
    text = re.sub(r'[^ㄱ-ㅎ가-힣a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    tokens = okt.pos(text)

    # 품사 선택
    selected_pos = ['Noun', 'Verb', 'Adjective']
    korean_words = [word for word, pos in tokens if pos in selected_pos and word not in korean_stopwords]

    # 영어 단어 추출
    english_words = re.findall(r'\b[a-z]+\b', text)
    english_words = [word for word in english_words if word not in english_stopwords]

    # 최종 단어 리스트 결합
    words = korean_words + english_words
    return ' '.join(words)

# 임베딩 생성 함수 정의
def get_embedding(text, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # 입력 데이터를 device로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs.get('token_type_ids', None),
            output_hidden_states=True,
            return_dict=True
        )
    # [CLS] 토큰의 임베딩 추출 후 CPU로 이동하여 numpy로 변환
    embedding = outputs['hidden_states'][-1][:, 0, :].detach().cpu().numpy()
    return embedding.squeeze()

# 적합도 계산 함수 정의
def calculate_fit(mentee, mentors, model):
    fit_scores = []

    # 멘티의 최신 bios_idx 기반 detail_content 전처리 및 임베딩 생성
    mentee_detail_content = mentee["detail_content"]
    mentee_detail_processed = preprocess_text(mentee_detail_content)
    mentee_embedding = get_embedding(mentee_detail_processed, model)

    # 최신 bios_idx 기준으로 멘토 리스트 정렬
    sorted_mentors = sorted(mentors, key=lambda x: x["bios_idx"], reverse=True)

    for mentor in sorted_mentors:
        # 멘토의 최신 bios_idx 기반 detail_content 전처리 및 임베딩 생성
        mentor_detail_content = mentor["detail_content"]
        mentor_detail_processed = preprocess_text(mentor_detail_content)
        mentor_embedding = get_embedding(mentor_detail_processed, model)

        # 유사도 계산
        similarity = cosine_similarity(
            [mentee_embedding],
            [mentor_embedding]
        )[0][0]

        fit_scores.append((mentor["user_name"], mentor["user_email"], similarity))

    # 멘토별 평균 유사도 계산
    averaged_fit_scores = {}
    for mentor_name, mentor_email, score in fit_scores:
        if mentor_name in averaged_fit_scores:
            averaged_fit_scores[mentor_name]["scores"].append(score)
        else:
            averaged_fit_scores[mentor_name] = {"email": mentor_email, "scores": [score]}

    averaged_fit_scores = [(mentor, data["email"], sum(data["scores"]) / len(data["scores"])) for mentor, data in averaged_fit_scores.items()]

    return sorted(averaged_fit_scores, key=lambda x: x[2], reverse=True)

# 데이터베이스 연결 정보 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'db': 'mentors',
    'charset': 'utf8mb4'
}

# 모델 로드 (앱 시작 시 한 번만 로드)
model = load_model(r'C:\Users\smhrd1\Desktop\Mentors\ai\models\KoELECTRA_base.pth')

# 멘티 및 멘토 데이터 조회 함수 정의
def get_mentee_mentor_data(mentee_email):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 멘티 데이터 조회
            sql_mentee = """
                SELECT u.user_email, u.user_name, u.user_category, u.mentor_yn, b.bios_idx, t.detail_idx, t.detail_content, t.parent_category_id, t.sub_category_name
                FROM tb_user u
                JOIN tb_bios b ON u.user_email = b.user_email
                JOIN tb_bios_detail t ON t.bios_idx = b.bios_idx
                WHERE u.mentor_yn = 'N' AND u.user_email = %s
                ORDER BY b.bios_idx DESC LIMIT 1;
            """
            cursor.execute(sql_mentee, (mentee_email,))
            mentee = cursor.fetchone()

            # 멘티가 없으면 None 반환
            if not mentee:
                return None, []

            # 멘토 데이터 조회 (멘티와 동일한 parent_category_id 및 sub_category_name을 가진 멘토들만)
            sql_mentors = """
                SELECT u.user_email, u.user_name, u.mentor_yn, b.bios_idx, t.detail_idx, t.detail_content, t.parent_category_id, t.sub_category_name
                FROM tb_user u
                JOIN tb_bios b ON u.user_email = b.user_email
                JOIN tb_bios_detail t ON t.bios_idx = b.bios_idx
                WHERE u.mentor_yn = 'Y' AND t.parent_category_id = %s AND t.sub_category_name = %s
                ORDER BY b.bios_idx DESC;
            """
            cursor.execute(sql_mentors, (mentee['parent_category_id'], mentee['sub_category_name']))
            mentors = cursor.fetchall()

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, []
    finally:
        conn.close()

    return mentee, mentors
