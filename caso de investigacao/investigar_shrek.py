import pandas as pd


ratings_df = pd.read_csv('ratings.csv')
ratings_df['date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

clusters_df = pd.read_csv('base utilizadas\clusters_resultado.csv')


movie_id = 4306


usuarios_cluster_104 = clusters_df[clusters_df['modularity_class'] == 44]['Id'].tolist()


toy_story1_ratings_cluster = ratings_df[
    (ratings_df['movieId'] == movie_id) &
    (ratings_df['userId'].isin(usuarios_cluster_104))
].copy()


print("Datas em que os usuários do cluster 44 avaliaram Shrek 1:")
print(toy_story1_ratings_cluster[['userId', 'date']])
