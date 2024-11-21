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
# Elimină filmele fără genuri
movies = movies[movies["genres"].map(len) > 0]  # Filme cu genuri valide
movies = movies[movies["keywords"].map(len) > 0]  # Filme cu keywords valide
movies = movies.reset_index(drop=True)
#
# print("numar total de filme dupa filtrare", movies.shape[0])
onehotencoding = MultiLabelBinarizer()
vector_genres = onehotencoding.fit_transform(movies["genres"])
onehotencondingkeywords=MultiLabelBinarizer()
keywords_vector= onehotencondingkeywords.fit_transform(movies["keywords"])
#
# print("dimnesiunea vector_genres", vector_genres.shape)
# print("dimensiunea keywords_vector", keywords_vector.shape)
assert vector_genres.shape[0] == keywords_vector.shape[0], "dimensiuni nealiniate"
marire_vector_genres =2* vector_genres
marire_vector_keywords= 1*keywords_vector
combined_vector = np.hstack([marire_vector_keywords,marire_vector_genres])
# Matricea de similaritate
matrice = cosine_similarity(combined_vector)
 # Funcția pentru recomandări
def recommend_movies(movie_title, top_n=5):
    try:
        movie_idx = movies[movies["title"] == movie_title].index[0]
    except IndexError:
        raise ValueError(f"Filmul '{movie_title}' nu a fost găsit.")

    similarities = matrice[movie_idx]
    similar_movies = pd.DataFrame({
        "title": movies["title"],
        "similarity": similarities,
        "genres":movies["genres"]
    })
    similar_movies = similar_movies[similar_movies.index != movie_idx]
    similar_movies = similar_movies.sort_values(by="similarity", ascending=False)
    return similar_movies.head(top_n)

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


#consola
if __name__ == "__main__":
    print("Recomandari filme bazate pe genuri si teme, Usercase1")
    while True:
        movie_title = input("Te rog introdu numele filmului pentru care doresti recomandarile sau scrie exit pentru a opri rularea:")
        if movie_title.lower() == "exit":
            print("Multumesc!Programul a fost închis.")
            break
        try:
            recommendations = recommend_movies(movie_title)
            print(f"Recomandarile pentru filmul cu titlu'{movie_title}'")
            print(recommendations)

        except ValueError as e:
            print(e)
            print("Te rog sa introduci alt nume de film\n")


#Usecase2
#SVD - pentru a gasi filme preferate de utilizatori asemanatori~ collaborative filter


