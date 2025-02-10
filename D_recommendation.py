import pandas as pd 
import numpy as np
from numpy.linalg import norm
import csv

#############################
# Carregamento dos Dados
#############################

def load_ratings(file_path):

    df = pd.read_csv(file_path, index_col=0)
    return df

def load_clusters(file_path):
    
    clusters = pd.read_csv(file_path)
    return clusters

def load_movies_metadata(file_path):
    
    rows = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # ["Id", "Name", "Genres"]
        for row in reader:
            if len(row) < 3:
                continue
            movie_id = row[0]
            genres = row[-1]
            
            name = ','.join(row[1:-1]).strip()
            rows.append({'Id': movie_id, 'Name': name, 'Genres': genres})
    metadata = pd.DataFrame(rows)
    metadata['Id'] = metadata['Id'].astype(str)
    return metadata

#############################
# Seleção do Cluster
#############################

def select_cluster_for_movie(ratings_df, clusters_df, filme_entrada, min_users=3, min_cluster_size=3, min_participation=0.3):
    
    cluster_scores = {}
    
    for cluster, grupo in clusters_df.groupby('modularity_class'):
        user_ids = grupo['Id'].tolist()
        if len(user_ids) < min_cluster_size:
            continue
        user_ids = [uid for uid in user_ids if uid in ratings_df.index]
        if not user_ids:
            continue
        cluster_size = len(user_ids)
        ratings_filme = ratings_df.loc[user_ids, filme_entrada]
        valid_ratings = ratings_filme[ratings_filme > 0]
        n_raters = valid_ratings.shape[0]
        if n_raters < min_users:
            continue
        participation = n_raters / cluster_size
        if participation < min_participation:
            continue
        avg_rating = valid_ratings.mean()
        reliability = avg_rating * participation
        cluster_scores[cluster] = {
            'avg_rating': avg_rating,
            'n_raters': n_raters,
            'cluster_size': cluster_size,
            'participation': participation,
            'reliability': reliability
        }
    
    if not cluster_scores:
        return None, None
    chosen_cluster = max(cluster_scores.items(), key=lambda x: x[1]['reliability'])[0]
    return chosen_cluster, cluster_scores[chosen_cluster]

#############################
# Similaridade entre Filmes (Rating)
#############################

def similaridade_cosseno(v1, v2):
    mask = (v1 > 0) & (v2 > 0)
    if np.sum(mask) == 0:
        return 0
    v1_filtrado = v1[mask]
    v2_filtrado = v2[mask]
    if norm(v1_filtrado) == 0 or norm(v2_filtrado) == 0:
        return 0
    return np.dot(v1_filtrado, v2_filtrado) / (norm(v1_filtrado) * norm(v2_filtrado))

#############################
# Similaridade de Gêneros Ponderada
#############################

def weighted_genre_similarity(movie_id_input, movie_id_candidate, metadata_df, genre_weights):
    
    try:
        input_row = metadata_df[metadata_df['Id'] == str(movie_id_input)]
        candidate_row = metadata_df[metadata_df['Id'] == str(movie_id_candidate)]
        if input_row.empty or candidate_row.empty:
            return 0
        genres_input = set(input_row.iloc[0]['Genres'].split('|'))
        genres_candidate = set(candidate_row.iloc[0]['Genres'].split('|'))
        intersection_weight = sum(genre_weights.get(g, 1) for g in genres_input.intersection(genres_candidate))
        union_weight = sum(genre_weights.get(g, 1) for g in genres_input.union(genres_candidate))
        if union_weight == 0:
            return 0
        return intersection_weight / union_weight
    except Exception as e:
        print("Erro no cálculo de similaridade de gêneros:", e)
        return 0

#############################
# Função de Recomendação (Definição de recommend_movies)
#############################

def recommend_movies(ratings_df, clusters_df, metadata_df, filme_entrada, chosen_cluster, 
                     intersec_threshold=0.7, similaridade_min=0.2, genre_threshold=0.4,
                     alpha=0.7, beta=0.3, genre_weights=None):
  
    if genre_weights is None:
        genre_weights = {}
    
    cluster_users = clusters_df[clusters_df['modularity_class'] == chosen_cluster]['Id'].tolist()
    cluster_users = [uid for uid in cluster_users if uid in ratings_df.index]
    cluster_size = len(cluster_users)
    if cluster_size == 0:
        return []
    
    vetor_entrada = ratings_df.loc[cluster_users, filme_entrada].values.astype(float)
    
    candidatos = []
    for filme in ratings_df.columns:
        if filme == filme_entrada:
            continue
        avaliacoes = ratings_df.loc[cluster_users, filme].values.astype(float)
        n_rated = np.sum(avaliacoes > 0)
        if n_rated < intersec_threshold * cluster_size:
            continue
        
        rating_sim = similaridade_cosseno(vetor_entrada, avaliacoes)
        if rating_sim < similaridade_min:
            continue
        
        media_avaliacoes = np.mean(avaliacoes[avaliacoes > 0])
        g_sim = weighted_genre_similarity(filme_entrada, filme, metadata_df, genre_weights)
        if g_sim < genre_threshold:
            continue
        
        combined_score = alpha * rating_sim + beta * g_sim
        candidatos.append((filme, media_avaliacoes, n_rated, rating_sim, g_sim, combined_score))
    
    candidatos = sorted(candidatos, key=lambda x: x[5], reverse=True)
    return candidatos

#############################
# Função Interativa Principal
#############################

def main():
    ratings_file = "base utilizadas/matriz_usuario_filme_10M.csv"
    clusters_file = "base utilizadas/clusters_resultado.csv"
    metadata_file = "base utilizadas/movies10M.csv"
    
    ratings_df = load_ratings(ratings_file)
    clusters_df = load_clusters(clusters_file)
    metadata_df = load_movies_metadata(metadata_file)
    
    # pesos para os gêneros
    genre_weights = {
        "Action": 0.8,
        "Adventure": 0.8,
        "Animation": 1.7,
        "Children": 1.2,
        "Comedy": 1.2,
        "Crime": 1.4,
        "Documentary": 1.3,
        "Drama": 1.3,
        "Fantasy": 1.4,
        "Film-Noir": 1.0,
        "Horror": 1.6,
        "IMAX": 0.3,
        "Musical": 1.5,
        "Mystery": 1.5,
        "Romance": 1.6,
        "Sci-Fi": 1.6,
        "Thriller": 1.5,
        "War": 1.6,
        "Western": 1.7
    }
    
    print("Digite 'sair' para encerrar o programa.")
    
    while True:
        filme_entrada = input("\nDigite o Id do filme de entrada: ").strip()
        if filme_entrada.lower() == "sair":
            print("Encerrando o programa.")
            break
        if filme_entrada not in ratings_df.columns:
            print("Filme não encontrado na base de dados.")
            continue
        
        chosen_cluster, info_cluster = select_cluster_for_movie(
            ratings_df, clusters_df, filme_entrada,
            min_users=6, min_cluster_size=6, min_participation=0.3
        )
        if chosen_cluster is None:
            print("Nenhum cluster com tamanho e participação suficientes avaliou o filme de entrada.")
            continue
        
        print(f"\nCluster escolhido: {chosen_cluster}")
        print("Informações do cluster:")
        print(f"  Média para o filme: {info_cluster['avg_rating']:.2f}")
        print(f"  Usuários que avaliaram: {info_cluster['n_raters']} de {info_cluster['cluster_size']}")
        print(f"  Participação: {info_cluster['participation']:.2f}")
        print(f"  Reliability score: {info_cluster['reliability']:.2f}")
        
        recomendacoes = recommend_movies(
            ratings_df, clusters_df, metadata_df, filme_entrada, chosen_cluster,
            intersec_threshold=0.7, similaridade_min=0.5, genre_threshold=0.4,
            alpha=0.7, beta=0.3, genre_weights=genre_weights
        )
        
        if not recomendacoes:
            print("Nenhum filme atende aos critérios de recomendação no cluster escolhido.")
        else:
            print("\nFilmes recomendados (MovieId, Média, Nº de avaliações, Rating_Sim, Genre_Sim, Combined_Score):")
            for movie, media, n_rated, r_sim, g_sim, comb in recomendacoes:
                nome = metadata_df.loc[metadata_df['Id'] == str(movie), 'Name'].values
                nome = nome[0] if nome.size > 0 else "Desconhecido"
                print(f"  Filme {movie} ({nome}): Média = {media:.2f} | Avaliações = {n_rated} | Rating_Sim = {r_sim:.2f} | Genre_Sim = {g_sim:.2f} | Score = {comb:.2f}")


main()