import unittest
from main import recommend_movies, movies, matrice
import numpy as np

class TestRecomandation(unittest.TestCase):
    @classmethod

    def setUp(cls):

        cls.movie_title = "Inception"
        if cls.movie_title not in movies["title"].tolist():
            raise ValueError(f"Movie '{cls.movie_title}' doesn't exist in the dataset")

    def test_no_self_recommendation(self):

        recommendations = recommend_movies(self.movie_title, top_n=5)
        recommended_titles = recommendations["title"].tolist()
        self.assertNotIn(
            self.movie_title,
            recommended_titles,
            f"The movie {self.movie_title} should not apper in its own recommendations."

        )

    def test_recommendation_order(self):

        recommendations = recommend_movies(self.movie_title, top_n=5)
        similarities = recommendations["similarity"].tolist()
        self.assertEqual(similarities, sorted(similarities, reverse=True), "TRecommendations are not sorted correctly")

    def test_recommendation_relevance(self):
        """Testăm relevanța recomandărilor pe baza genurilor."""
        movie_genres = set(movies[movies["title"] == self.movie_title]["genres"].iloc[0])
        recommendations = recommend_movies(self.movie_title, top_n=5)
        for _, row in recommendations.iterrows():
            recommended_genres = set(row["genres"])
            self.assertTrue(movie_genres & recommended_genres,f"The movie {row['title']} does not share genres with {self.movie_title}.")
    def test_recommendation_limit(self):

        top_n=4
        recommendations = recommend_movies(self.movie_title, top_n=top_n)
        self.assertEqual(len(recommendations), top_n, "Expected {top_n} recommenations, but but {len(recommendations)}")



if __name__ == "__main__":
    unittest.main()
