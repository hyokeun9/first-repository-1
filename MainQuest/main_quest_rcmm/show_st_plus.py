import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autointmlp import AutoIntMLPModel, predict_model

# =========================
# 데이터 & 모델 로드
# =========================
@st.cache_resource
def load_data():
    '''
    앱에서 보여줄 필요 데이터를 가져오는 함수입니다.
    - 사용자, 영화, 평점 데이터를 가져옵니다.
    - 앞서 저장된 모델도 불러오고 구현해둡니다.
    '''
    project_path = os.path.abspath(os.getcwd())
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = f"{project_path}/{data_dir_nm}"
    model_path = f"{project_path}/{model_dir_nm}"
    
    field_dims = np.load(f'{data_path}/field_dims.npy')
    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
    users_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')
    
    # AutoIntMLP 모델 초기화 및 가중치 로드
    model = AutoIntMLPModel(field_dims, embed_dim=16, att_layer_num=3, att_head_num=2,
                            att_res=True, dnn_hidden_units=(32,32), dnn_activation='relu',
                            l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0.3, init_std=0.0001)
    model(tf.constant([[0]*len(field_dims)], dtype=tf.int32))
    model.load_weights(f'{model_path}/autoIntMLP_model_weights.weights.h5')
    
    # 학습시 사용한 label encoder 불러오기
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    return users_df, movies_df, ratings_df, model, label_encoders

users_df, movies_df, ratings_df, model, label_encoders = load_data()

# =========================
# 추천 관련 함수
# =========================
def get_user_seen_movies(ratings_df):
    '''
    사용자가 과거에 보았던 영화 리스트를 가져옵니다.
    '''
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seen_dict(movies_df, users_df, user_seen_movies):
    '''
    사용자가 보지 않았던 영화 리스트를 가져옵니다.
    '''
    unique_movies = movies_df['movie_id'].unique()
    user_non_seen_dict = {}
    for user in users_df['user_id'].unique():
        seen = user_seen_movies[user_seen_movies['user_id']==user]['movie_id'].values[0]
        user_non_seen_dict[user] = list(set(unique_movies) - set(seen))
    return user_non_seen_dict

def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
    '''
    추천 결과를 가져오는 함수입니다.
    1. 사용자가 보지 않은 영화 리스트 가져오기
    2. 모델 입력용 데이터프레임 구성
    3. 라벨 인코딩 적용
    4. 모델 predict 후 원본 영화 id로 변환
    '''
    user_non_seen_movie = user_non_seen_dict.get(user)
    user_id_list = [user]*len(user_non_seen_movie)
    r_decade = str(r_year - (r_year%10)) + 's'
    
    # 영화 정보와 사용자 정보 합치기
    user_non_seen_movie = pd.merge(pd.DataFrame({'movie_id':user_non_seen_movie}), movies_df, on='movie_id')
    user_info = pd.merge(pd.DataFrame({'user_id':user_id_list}), user_df, on='user_id')
    user_info['rating_year'] = r_year
    user_info['rating_month'] = r_month
    user_info['rating_decade'] = r_decade
    
    merge_data = pd.concat([user_non_seen_movie, user_info], axis=1)
    merge_data.fillna('no', inplace=True)
    merge_data = merge_data[['user_id','movie_id','movie_decade','movie_year','rating_year','rating_month','rating_decade',
                             'genre1','genre2','genre3','gender','age','occupation','zip']]
    
    # 라벨 인코딩 적용
    for col, le in label_encoders.items():
        known = set(le.classes_)
        fallback = le.classes_[0]
        merge_data[col] = merge_data[col].apply(lambda x: x if x in known else fallback)
        merge_data[col] = le.transform(merge_data[col])
    
    # 모델 predict 후 추천 영화 추출
    recom_top = predict_model(model, merge_data)
    recom_top = [r[0] for r in recom_top]
    origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    return movies_df[movies_df['movie_id'].isin(origin_m_id)]

# =========================
# UI 구성
# =========================
st.title("🎬 영화 추천 결과 살펴보기")

user_seen_movies = get_user_seen_movies(ratings_df)
user_non_seen_dict = get_user_non_seen_dict(movies_df, users_df, user_seen_movies)

st.header("사용자 정보 입력")
col1, col2, col3 = st.columns(3)
with col1:
    user_id = st.number_input("사용자 ID", min_value=users_df['user_id'].min(),
                              max_value=users_df['user_id'].max(), value=users_df['user_id'].min())
with col2:
    r_year = st.slider("추천 타겟 연도", min_value=ratings_df['rating_year'].min(),
                       max_value=ratings_df['rating_year'].max(), value=ratings_df['rating_year'].min())
with col3:
    r_month = st.selectbox("추천 타겟 월", options=list(range(1,13)), index=0)

if st.button("추천 결과 보기"):
    st.subheader("사용자 기본 정보")
    st.dataframe(users_df[users_df['user_id']==user_id])
    
    st.subheader("추천 영화 Top 10")
    recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders)
    
    # 카드 느낌으로 5개씩 나눠서 표시
    cols = st.columns(5)
    for i, movie in enumerate(recommendations.head(10).itertuples()):
        with cols[i%5]:
            st.text(f"{movie.title}\n{movie.genre1}/{movie.genre2}")


