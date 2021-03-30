import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Print the first three rows
metadata.head(3)

C = metadata['vote_average'].mean()
# print(C)

m = metadata['vote_count'].quantile(0.90)
# print(m)

q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
# print(q_movies.shape)

# print(metadata.shape)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)


#Print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(100))

# print(metadata['overview'].head())

from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
# print(tfidf_matrix.shape)
#
# print(tfidf.get_feature_names()[5000:5010])
#
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# print(cosine_sim.shape)
#
# print(cosine_sim[1])

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# print(indices[:10])

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

a = input(str("Input: "))
print("--------------------------- Out put !!! ----------------------------")

print(get_recommendations(a))

