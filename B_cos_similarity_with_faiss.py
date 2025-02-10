import numpy as np
import pandas as pd
import time
import faiss
from tqdm import tqdm

def ler_dados_dense(file_path):
    
    df = pd.read_csv(file_path, index_col=0)
    df.fillna(0, inplace=True)
    ratings = df.values.astype('float32')
    # Garante que o array é contíguo em memória
    ratings = np.ascontiguousarray(ratings)
    return ratings, df.index, df.columns

def faiss_similarity_search(ratings_dense, similarity_threshold=0.70, k=100):
    
    # Normaliza cada linha
    faiss.normalize_L2(ratings_dense)
    
    d = ratings_dense.shape[1]  # dimensão dos vetores (número de filmes)
    index = faiss.IndexFlatIP(d)  # índice que utiliza produto interno (IP)
    index.add(ratings_dense)
    
   
    distances, indices_nn = index.search(ratings_dense, k)
    
    pairs = []
    num_users = ratings_dense.shape[0]
    
    for i in tqdm(range(num_users), desc="Processando usuários", unit="usuário"):
        for j, sim in zip(indices_nn[i], distances[i]):
            # Ignora o próprio usuário e duplicatas (i < j)
            if j == i or j < i:
                continue
            if sim >= similarity_threshold:
                pairs.append((i, j, sim))
    return pairs

def exportar_para_csv(pares_similares, indices, output_file="faiss_edges.csv"):
   
    data = []
    for i, j, sim in pares_similares:
        data.append({
            "Source": indices[i],
            "Target": indices[j],
            "Weight": round(sim, 3)
        })
    df_edges = pd.DataFrame(data, columns=["Source", "Target", "Weight"])
    df_edges.to_csv(output_file, index=False)
    print(f"Arquivo CSV exportado para {output_file}")

def main():
    start_time = time.time()
    
    arquivo_csv = "base utilizadas/matriz_usuario_filme_10M.csv"
    print("Lendo a matriz de avaliações (densa)...")
    ratings_dense, indices, colunas = ler_dados_dense(arquivo_csv)
    
    print("Executando busca de similaridade com FAISS...")
    pares_similares = faiss_similarity_search(ratings_dense, similarity_threshold=0.60, k=100)
    
    print("Exportando resultados...")
    exportar_para_csv(pares_similares, indices, output_file="base utilizadas/faiss_GRAFOS_10M(70S).csv")
    
    elapsed = time.time() - start_time
    print(f"Tempo total de execução: {elapsed:.2f} segundos")


main()