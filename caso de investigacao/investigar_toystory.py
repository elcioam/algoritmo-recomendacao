import pandas as pd

ratings_df = pd.read_csv('ratings.csv')
ratings_df['date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

clusters_df = pd.read_csv('base utilizadas\clusters_resultado.csv')


toy_story1_id = 1


usuarios_cluster_104 = clusters_df[clusters_df['modularity_class'] == 104]['Id'].tolist()

toy_story1_ratings_cluster = ratings_df[
    (ratings_df['movieId'] == toy_story1_id) &
    (ratings_df['userId'].isin(usuarios_cluster_104))
].copy()

# Exibe a lista de usuários e as datas em que eles avaliaram Toy Story 1
print("Datas em que os usuários do cluster 104 avaliaram Toy Story 1:")
print(toy_story1_ratings_cluster[['userId', 'date']])
