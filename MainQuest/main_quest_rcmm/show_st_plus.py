import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autointmlp import AutoIntMLPModel, predict_model

# 1. 페이지 설정 (가장 상단)
st.set_page_config(page_title="Movie RecSys", page_icon="🎬", layout="wide")

# 2. 데이터 및 모델 로드 함수 (캐싱 적용)
@st.cache_resource
def load_data():
    project_path = os.path.abspath(os.getcwd())
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = os.path.join(project_path, data_dir_nm)
    model_path = os.path.join(project_path, model_dir_nm)
    
    field_dims = np.load(f'{data_path}/field_dims.npy')
    dropout = 0.3
    embed_dim = 16
    
    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
    user_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')

    model = AutoIntMLPModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True, 
                            dnn_hidden_units=(32, 32), dnn_dropout=dropout)
    
    # 더미 데이터로 모델 빌드 (weights 로드 전 필수)
    model(tf.constant([[0] * len(field_dims)], dtype=tf.int32))
    model.load_weights(f'{model_path}/autoIntMLP_model_weights.weights.h5') 
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    
    return user_df, movies_df, ratings_df, model, label_encoders

# 3. 비즈니스 로직 함수들 (사전 정의)
def get_user_seen_movies(ratings_df):
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seed_dict(movies_df, user_df, user_seen_movies):
    unique_movies = movies_df['movie_id'].unique()
    unique_users = user_df['user_id'].unique()
    user_non_seen_dict = {}
    for user in unique_users:
        user_seen_movie_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
        user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_movie_list))
        user_non_seen_dict[user] = user_non_seen_movie_list
    return user_non_seen_dict

def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
    user_non_seen_movie = user_non_seen_dict.get(user)
    user_id_list = [user for _ in range(len(user_non_seen_movie))]
    r_decade = str(r_year - (r_year % 10)) + 's'
    
    user_non_seen_movie_df = pd.merge(pd.DataFrame({'movie_id':user_non_seen_movie}), movies_df, on='movie_id')
    user_info = pd.merge(pd.DataFrame({'user_id':user_id_list}), user_df, on='user_id')
    user_info['rating_year'] = r_year
    user_info['rating_month'] = r_month
    user_info['rating_decade'] = r_decade
    
    merge_data = pd.concat([user_non_seen_movie_df, user_info], axis=1)
    merge_data.fillna('no', inplace=True)
    merge_data = merge_data[['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']]
    
    for col, le in label_encoders.items():
        known = set(le.classes_)
        fallback = le.classes_[0]
        merge_data[col] = merge_data[col].apply(lambda x: x if x in known else fallback)
        merge_data[col] = le.transform(merge_data[col])
    
    recom_top = predict_model(model, merge_data)
    recom_top = [r[0] for r in recom_top]
    origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    origin_m_id = origin_m_id.astype(movies_df['movie_id'].dtype)
    
    return movies_df[movies_df['movie_id'].isin(origin_m_id)]

# 4. 데이터 실행 (메인 로직)
with st.spinner('데이터를 불러오고 있습니다...'):
    users_df, movies_df, ratings_df, model, label_encoders = load_data()
    user_seen_movies = get_user_seen_movies(ratings_df)
    user_non_seen_dict = get_user_non_seed_dict(movies_df, users_df, user_seen_movies)

# 5. UI 레이아웃
st.title("🎬 Movie Recommendation System")
st.markdown("AutoInt+ 모델을 활용하여 사용자의 취향에 맞는 영화를 추천합니다.")

# 사이드바 입력창
with st.sidebar:
    st.header("⚙️ 개인화 설정")
    user_id = st.number_input("👤 사용자 ID 입력", 
                               min_value=int(users_df['user_id'].min()), 
                               max_value=int(users_df['user_id'].max()), 
                               value=int(users_df['user_id'].min()))
    
    st.subheader("📅 추천 시점 설정")
    r_year = st.slider("연도(Year)", int(ratings_df['rating_year'].min()), int(ratings_df['rating_year'].max()), 2000)
    r_month = st.slider("월(Month)", 1, 12, 1)
    
    predict_btn = st.button("🚀 추천 결과 생성", use_container_width=True)

# 결과 표시 구역
if predict_btn:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 사용자 정보")
        st.table(users_df[users_df['user_id'] == user_id])
        
    with col2:
        st.subheader("🍿 높은 평점을 줬던 영화 (Top 5)")
        past_interactions = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')
        st.dataframe(past_interactions[['movie_id', 'genre1', 'rating']].head(5), use_container_width=True)

    st.divider()
    
    st.subheader("🌟 추천 결과 (Top-N)")
    with st.status("인공지능 모델이 영화를 고르고 있습니다...", expanded=True) as status:
        recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders)
        status.update(label="추천 완료!", state="complete", expanded=False)
    
    st.balloons()
    st.dataframe(recommendations, use_container_width=True)
else:
    st.info("👈 왼쪽 사이드바에서 사용자 ID를 선택하고 버튼을 눌러주세요!")