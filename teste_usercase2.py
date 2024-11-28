import unittest
from main import recommend_from_similar_users, find_similar_users, ratings, user_features, train, model, movie_titles
from surprise import Reader, Dataset

ratings = ratings.groupby('movieId').filter(lambda x: len(x) >= 10)

class TestUsercase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(ratings.head())
        print(ratings.columns)
        print(ratings["userId"].unique())
        cls.user_id = 2
        if cls.user_id not in ratings["userId"].unique():
            raise ValueError(f"User id '{cls.user_id}' does not exist in dataset")

    def test_existing_user_id(self):
        self.assertIn(self.user_id, ratings["userId"].tolist(), "This user should exist in the dataset")

    def test_find_similar_users(self):
        similar_users, similarities = find_similar_users(self.user_id, user_features, train, k=5)
        self.assertEqual(len(similar_users), 5, "The function should return  5 similar users")

    def test_recommend_from_similar_users(self):
        similar_users, _ = find_similar_users(self.user_id, user_features, train, k=5)
        recommendations = recommend_from_similar_users(similar_users, train, self.user_id, movie_titles, model, top_n=5)
        self.assertEqual(len(recommendations), 5, "The function should return  5 recommendations")

    def test_missing_data(self):
        incomplete_ratings = ratings.drop(columns=["movieId"])
        with self.assertRaises(KeyError, msg="Missing data should raise a KeyError"):
            Dataset.load_from_df(incomplete_ratings[["userId", "rating"]], Reader(rating_scale=(0.5, 5)))

    def test_similarity_consistency(self):
        similar_users, similarities = find_similar_users(self.user_id, user_features, train, k=5)
        self.assertTrue(
            all(similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1)),
            "Similarities should be in descending order"
        )

if __name__ == "__main__":
    unittest.main()
