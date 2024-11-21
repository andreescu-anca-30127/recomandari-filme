import unittest
from main import recommend_movies, movies, matrice
import numpy as np

class TestRecomandation(unittest.TestCase):

    def setUp(self):
        self.movie_title = input ("introdu filmul pt testare:")
        if self.movie_title not in movies["title"].tolist():
            self.fail("filmul nu exista in baza de date")

    def test_no_self_recommendation(self):
        recommendations = recommend_movies(self.movie_title, top_n=5)
        recommended_titles = recommendations["title"].tolist()
        self.assertNotIn(
            self.movie_title,
            recommended_titles,
            f"{self.movie_title} nu ar trebui să apară în recomandări."

        )

    def test_recommendation_order(self):
        """Testăm dacă recomandările sunt ordonate descrescător după similaritate."""
        recommendations = recommend_movies(self.movie_title, top_n=5)
        similarities = recommendations["similarity"].tolist()
        self.assertEqual(similarities, sorted(similarities, reverse=True), "Recomandările nu sunt ordonate corect.")

    def test_recommendation_relevance(self):
        """Testăm relevanța recomandărilor pe baza genurilor."""
        movie_genres = set(movies[movies["title"] == self.movie_title]["genres"].iloc[0])
        recommendations = recommend_movies(self.movie_title, top_n=5)
        for _, row in recommendations.iterrows():
            recommended_genres = set(row["genres"])
            self.assertTrue(movie_genres & recommended_genres,f"Filmul {row['title']} nu are genuri comune cu {self.movie_title}.")


