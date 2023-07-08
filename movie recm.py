import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load movie ratings data
ratings_data = pd.read_csv('ratings.csv')  
movies_data = pd.read_csv('movies.csv')  

# Merge ratings and movie data
data = pd.merge(ratings_data, movies_data, on='movieId')

# Create a pivot table of user ratings
user_ratings = data.pivot_table(index='userId', columns='title', values='rating')

# Compute similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(user_ratings.fillna(0))

# Function to recommend movies
def recommend_movies(user_id, num_recommendations=5):
    # Get the index of the user in the similarity matrix
    user_index = user_ratings.index.get_loc(user_id)

    # Get similarity scores for the user
    user_similarity_scores = similarity_matrix[user_index]

    # Sort movies based on similarity scores
    sorted_indices = user_similarity_scores.argsort()[::-1]

    # Get top n similar users
    top_similar_users = sorted_indices[1:num_recommendations + 1]

    # Get movies rated by similar users
    movies_rated_by_similar_users = user_ratings.iloc[top_similar_users].dropna(axis=1)

    # Calculate average rating for each movie
    average_ratings = movies_rated_by_similar_users.mean()

    # Get movies not rated by the user
    user_movies = user_ratings.iloc[user_index].dropna().index
    unrated_movies = set(average_ratings.index) - set(user_movies)

    # Sort unrated movies by average rating
    recommended_movies = average_ratings.loc[unrated_movies].sort_values(ascending=False).head(num_recommendations)

    return recommended_movies

# Example usage
user_id = 1
recommendations = recommend_movies(user_id)
print(f"Recommendations for User {user_id}:")
print(recommendations)