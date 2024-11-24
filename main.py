# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

pd.set_option('display.max.colwidth', None)
pd.set_option('display.max_rows', None)#randuri
pd.set_option('display.max_columns', None)#coloanae

moviesdata = pd.read_csv(r'C:\movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(r'C:\ratings.csv', low_memory=False)
keywords = pd.read_csv(r'C:\keywords.csv', low_memory=False)

print(moviesdata.columns)
print(keywords.columns)



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
print("a facut functia de extragere de genuri")
# Elimină filmele fără genuri
movies = movies[movies["genres"].map(len) > 0]  # Filme cu genuri valide
movies = movies[movies["keywords"].map(len) > 0]  # Filme cu keywords valide
movies = movies.reset_index(drop=True)
#
# # print("numar total de filme dupa filtrare", movies.shape[0])
# onehotencoding = MultiLabelBinarizer()
# vector_genres = onehotencoding.fit_transform(movies["genres"])
# onehotencondingkeywords=MultiLabelBinarizer()
# keywords_vector= onehotencondingkeywords.fit_transform(movies["keywords"])
#
# print("dimnesiunea vector_genres", vector_genres.shape)
# print("dimensiunea keywords_vector", keywords_vector.shape)
# assert vector_genres.shape[0] == keywords_vector.shape[0], "dimensiuni nealiniate"
# marire_vector_genres =2* vector_genres
# marire_vector_keywords= 1*keywords_vector
# combined_vector = np.hstack([marire_vector_keywords,marire_vector_genres])
# Matricea de similaritate
# matrice = cosine_similarity(combined_vector)
 # Funcția pentru recomandări
# def recommend_movies(movie_title, top_n=5):
#     try:
#         movie_idx = movies[movies["title"] == movie_title].index[0]
#     except IndexError:
#         raise ValueError(f"Filmul '{movie_title}' nu a fost găsit.")
#
#     similarities = matrice[movie_idx]
#     similar_movies = pd.DataFrame({
#         "title": movies["title"],
#         "similarity": similarities,
#         "genres":movies["genres"]
#     })
#     similar_movies = similar_movies[similar_movies.index != movie_idx]
#     similar_movies = similar_movies.sort_values(by="similarity", ascending=False)
#     return similar_movies.head(top_n)

#testare
# recommendations = recommend_movies("Inception", top_n=5)
# print(recommendations)
# idx_inception = movies[movies["title"] == "Inception"].index[0]
# idx_paycheck = movies[movies["title"] == "Paycheck"].index[0]

# # verificare personaa
# print("Vector genuri pentru Inception:", vector_genres[idx_inception])
# print("Vector genuri pentru Paycheck:", vector_genres[idx_paycheck])
#
# # Compara  vectorii
# print("Sunt vectorii identici?", (vector_genres[idx_inception] == vector_genres[idx_paycheck]).all())


#Usecase2
#SVD - pentru a gasi filme preferate de utilizatori asemanatori~ collaborative filter
print("intra in svd")
reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(ratings[['userId', 'movieId','rating']], reader)
train, test =train_test_split(data, test_size=0.2)
model = SVD()
model.fit(train)
print("a antrenat modeluk")
def svd(user_id, n=10):
    alreadyseen_movie= ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all = ratings['movieId'].unique()
    unseen_movies = [movie for movie in all if movie not in alreadyseen_movie]
    predictions = [model.predict(user_id, movie) for movie in unseen_movies]

    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    # Selectează primele N recomandări
    top_recommendations = [(pred.iid, pred.est) for pred in predictions[:n]]

    return top_recommendations
def precision(user_id, suggestion, s=10):
    relevant_movies = ratings[(ratings['userId'] == user_id) & (ratings['rating'] > 3.5)]['movieId'].tolist()
    relevant_movies_set= set(relevant_movies)
    print(f"filme relevante pentru user {relevant_movies}")
    recommended_movies = [movie_id for movie_id, _ in suggestion[:s]]
    print(f"filme recomndate {recommended_movies}")

    relevant_in_suggestion =set(recommended_movies) & relevant_movies_set
    print(f"filme relevante in sugestii {relevant_in_suggestion}")
    precision= len(relevant_in_suggestion)/s  if s>0 else 0
    return precision

#
# user_id=2
# top_reco = svd (user_id, n=10)
# print(f"recomandari pe baza de user:{top_reco}")

# #consola
if __name__ == "__main__":
    print("Recomandari filme bazate pe genuri si teme sau pe recomandarile altor useri")
    print("Usercase1: Recomandari filme pentru un film dat, bazate pe teme si genuri")
    print("Usercase2: Recomandari pentru un anume utilizator")
    print("Exit: Tastati exit pentru parasirea programului")
    while True:
        your_choise = input("Te rog sa iti alegi optiunea Usercase1/Usercase2/Exit:")
        if your_choise == "Usercase1":  # Adăugat `:`
            movie_title = input("Te rog introdu numele filmului pentru care doresti recomandarile: ")
        #     try:
        #         recommendations = recommend_movies(movie_title)
        #         print(f"Recomandarile pentru filmul cu titlu '{movie_title}' sunt:")
        #
        #
        #     except ValueError as e:
        #         print(e)
        #         print("Te rog sa introduci alt nume de film.\n")

        elif your_choise == "Usercase2":

            try:

                user_id = int(input("Introdu id-ul utilizatorului pentru care doresti sa se realizeze recomandarea:"))
                top_reco = svd(user_id, n=10)
                print("Recomandarile personalizate pentru utilizatorul ales sunt: ")
                for title, score in top_reco:
                    print(f"{title} - Scor: {score:.2f}")
                precision = precision(user_id,top_reco,s=10)
                print(f"precizia este:{precision:.2f}")
            except ValueError as e:
                    print(e)
                    print("te rog sa introduci un id corect:")
        elif your_choise =="Exit":
            print("Multumesc!Programu urmeaza sa se inchida")
            break
        else:
            print("Ati ales o optiune gresita va rog sa alegeti alta:")

