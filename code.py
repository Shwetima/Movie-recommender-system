# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv("C:\\Users\\hp\\Python36\\tmdb_5000_movies.csv")

# Print the first three rows
#print (metadata.head(3))

# Calculate C
C = metadata['vote_average'].mean()
print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print(m)

# Filter out all qualified movies into a new DataFrame and any change made to q_movies won't affect the data of movies stored in main database
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
#define the matrix size of the qualified movies using q_movies.shape
q_movies.shape 

#function to calculate weighted rating
def weigthed_rating(x, m=m, C=C):
	v=x['vote_count']
	R=x['vote_average']
	return (v/(m+v)*R +m/(m+v)*C)

#defining a new feature into data frame named score
q_movies['score']=q_movies.apply(weigthed_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending =False)

#Print the top 15 movies
#print (q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))

#print(metadata['overview'].head())

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#print (tfidf_matrix)

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
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

print (get_recommendations('The Dark Knight Rises'))
	
	