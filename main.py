# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import KNNBasic

pd.set_option('display.max.colwidth', None)
pd.set_option('display.max_rows', None)#randuri
pd.set_option('display.max_columns', None)#coloanae

moviesdata = pd.read_csv(r'C:\movies_metadata.csv', low_memory=False)
keywords = pd.read_csv(r'C:\keywords.csv', low_memory=False)
# print(moviesdata.columns)
# print(keywords.columns)


# Convert to Parquet - for storing dataset in compact format
ratings = pd.read_csv(r'C:\ratings.csv', low_memory=False)
ratings.to_parquet('ratings.parquet')

# Load the dataset when needed
ratings = pd.read_parquet('ratings.parquet')
ratings = ratings.groupby('movieId').filter(lambda x: len(x) >= 10)

movie_titles = dict(zip(moviesdata['id'], moviesdata['title']))

# # Elimină rândurile cu valori NaN sau nevalide în coloana 'id'
moviesdata = moviesdata[moviesdata['id'].notna()]
moviesdata = moviesdata[moviesdata['id'].str.isnumeric()]
moviesdata['id'] = moviesdata['id'].astype(int)

# #USERCASE1
def extract_keywords(keywords_column):
    try:
        keywords = ast.literal_eval(keywords_column)
        return[k['name'] for k in keywords]
    except:
        return []
    keywords["keywords"] = keywords["keywords"].apply(extract_keywords)
moviesdata = moviesdata.merge(keywords, left_on="id", right_on="id", how="left")
moviesdata['keywords'] = moviesdata['keywords'].fillna("[]")

movies = moviesdata[["title", "genres", "keywords"]].dropna().reset_index(drop=True)

def extract_genres(genres_column):

    try:
        genres = ast.literal_eval(genres_column)
        return [g["name"] for g in genres]
    except:
        return []
# # Aplică funcția de preprocesare
movies["genres"] = movies["genres"].apply(extract_genres)
#
print("The genres extractions was performed")
# remove movies without genres
movies = movies[movies["genres"].map(len) > 0]  # Filme cu genuri valide
movies = movies[movies["keywords"].map(len) > 0]  # Filme cu keywords valide
movies = movies.reset_index(drop=True)
#MultiLabelBinarizer
onehotencoding = MultiLabelBinarizer()
vector_genres = onehotencoding.fit_transform(movies["genres"])
onehotencondingkeywords=MultiLabelBinarizer()
keywords_vector= onehotencondingkeywords.fit_transform(movies["keywords"])
print("dimnesiunea vector_genres", vector_genres.shape)
print("dimensiunea keywords_vector", keywords_vector.shape)
#increase the importance
marire_vector_genres = 2 * vector_genres
marire_vector_keywords = 1*keywords_vector
combined_vector = np.hstack([marire_vector_keywords,marire_vector_genres])
matrice = cosine_similarity(combined_vector)

def recommend_movies(movie_title, top_n=5):
    try:
        movie_idx = movies[movies["title"] == movie_title].index[0]
    except IndexError:
        raise ValueError(f" The movie'{movie_title}' was not found.")

    similarities = matrice[movie_idx]
    similar_movies = pd.DataFrame({
        "title": movies["title"],
        "similarity": similarities,
        "genres":movies["genres"]
    })
    similar_movies = similar_movies[similar_movies.index != movie_idx]
    similar_movies = similar_movies.sort_values(by="similarity", ascending=False)
    return similar_movies.head(top_n)

#manual testing
recommendations = recommend_movies("Babylon A.D.", top_n=5)
print(recommendations)
idx_inception = movies[movies["title"] == "Inception"].index[0]
idx_paycheck = movies[movies["title"] == "Paycheck"].index[0]

# verificare personala
print("Vector genuri pentru Inception:", vector_genres[idx_inception])
print("Vector genuri pentru Paycheck:", vector_genres[idx_paycheck])

# Compara  vectorii
print("Sunt vectorii identici?", (vector_genres[idx_inception] == vector_genres[idx_paycheck]).all())


#Usecase2

reader = Reader(rating_scale=(0.5,5)) #evaluare de utilizator de la 0.5 la 5
data = Dataset.load_from_df(ratings[['userId', 'movieId','rating']], reader)
train, test =train_test_split(data, test_size=0.2)
#SVD
model = SVD()
model.fit(train)
prediction = model.test(test)
print("RMSE:", accuracy.rmse(prediction))
print("MAE:", accuracy.mae(prediction))
def get_user_features(model, train):

    n_users = train.n_users
    user_features = np.array([model.pu[i] for i in range(n_users)])
    return user_features


user_features = get_user_features(model, train)

def find_similar_users(user_id, user_features, train, k=5):

    inner_user_id = train.to_inner_uid(user_id)
    target_user_features = user_features[inner_user_id].reshape(1, -1)
    similarities = cosine_similarity(target_user_features, user_features).flatten()
    similar_user_indices = np.argsort(-similarities)[1:k+1]
    return similar_user_indices, similarities[similar_user_indices]

def recommend_from_similar_users(similar_users, train, user_id, movie_titles, model, top_n=10):
    # Movies already rated by the target user
    inner_user_id = train.to_inner_uid(user_id)
    rated_movies = {movie_id for movie_id, _ in train.ur[inner_user_id]}


    recommended_movies = {}
    for sim_user in similar_users:
        user_ratings = train.ur[sim_user]
        for movie_id, rating in user_ratings:
            if movie_id not in rated_movies:
                predicted_rating = model.predict(user_id, train.to_raw_iid(movie_id)).est
                if movie_id not in recommended_movies:
                    recommended_movies[movie_id] = predicted_rating
                else:
                    recommended_movies[movie_id] += predicted_rating

    # Sort movies by predicted rating
    recommendations = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations

#Consola
if __name__ == "__main__":

    print("Movie recmmendations based on genres, themes or other users suggestions")
    print("Usecase1: Reccomendations for a given movies, based on themes and genres")
    print("Usecase2: Reccomendations for a specific user")
    print("Exit: Type Exit to leave the program")

    while True:
        your_choise = input("Please select your option  Usercase1/Usercase2/Exit:")
        if your_choise == "Usercase1":
            movie_title = input("Please enter the name of the movie for which you want recommendations:")
            try:
                recommendations = recommend_movies(movie_title)
                print(f"The recommendations for the movie titled '{movie_title}' are:")
                for index, row in recommendations.iterrows():
                    print(
                        f"Title: {row['title']}, Similarity: {row['similarity']:.2f}, Genres: {', '.join(row['genres'])}")



            except ValueError as e:
                print(e)
                print("Please enter another movie name.\n")

        elif your_choise == "Usercase2":

            try:

                user_id = int(input("Please enter the user ID for which the recommendation should be made:"))
                if user_id not in ratings['userId'].unique():
                    raise ValueError(f"The user ID {user_id} does not exist in the dataset.")

                # Find similar users
                similar_users, similarities = find_similar_users(user_id, user_features, train, k=5)

                # Recommend based on similar users
                top_reco = recommend_from_similar_users(similar_users, train, user_id, movie_titles, model, top_n=10)

                print("The personalized recommendations for the selected user are:")
                for movie_id, score in top_reco:
                    print(f"{movie_id} - Scor: {score:.2f}")


            except ValueError as e:
                    print(e)
                    print("Please enter a valid user:")

        elif your_choise =="Exit":
            print("Thank you! The program is about to close")
            break
        else:
            print("You have selected an incorrect option. Please choose another one")

#SVD model
# sim_options = {
#     "name": "cosine",
#     "user_based": True,  #user-based collaborative filtering
# }
# model=KNNBasic(sim_options=sim_options, k=20)
# model.fit(train)
# def get_recommendations(model,train, user_id, s=10):
#     inner_user_id = train.to_inner_uid(user_id)
#
#     similar_users = model.get_neighbors(inner_user_id, k=10)
#     recommended_movies = {}
#
#     for similar_user in similar_users:
#         user_ratings = train.ur[similar_user]
#         for movie_id, rating in user_ratings:
#             if movie_id  not in train.ur[inner_user_id]:
#                 if movie_id not in recommended_movies:
#                     recommended_movies[movie_id] = rating
#                 else:
#                     recommended_movies[movie_id] += rating
#
#     sorted_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
#     top_n_movies = sorted_movies[:s]
#     top_n_movies = [
#         (movie_titles.get(int(train.to_raw_iid(movie_id)), "Unknown Title"), score)
#         for movie_id, score in top_n_movies
#     ]
#     return top_n_movies
# def get_recommendations_svd(model, user_id, train, top_n=10):
#     all_movie_ids= set(train.all_items())
#     rated_movies = {movie_id for movie_id, _ in train.ur[train.to_inner_uid(user_id)]}
#     recommendations =[]
#     for movie_id in all_movie_ids -rated_movies:
#         est_rating= model.predict(user_id,train.to_raw_iid(movie_id)).est
#         recommendations.append((movie_id,est_rating))
#     recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
#
#
#
#     return recommendations
