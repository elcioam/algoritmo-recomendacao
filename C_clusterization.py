import pandas as pd
import networkx as nx
import community as community_louvain  # precisa do pip install python-louvain

def load_edges(file_path):
    
    try:
        df = pd.read_csv(file_path)
        print("Dados carregados com sucesso.")
        return df
    except Exception as e:
        print("Erro ao carregar o arquivo:", e)
        return None

def create_graph_from_edges(df):
    
    G = nx.Graph()
    for _, row in df.iterrows():
        source = row['Source']
        target = row['Target']
        weight = row['Weight']
        G.add_edge(source, target, weight=weight)
    print("Grafo criado com", G.number_of_nodes(), "nós e", G.number_of_edges(), "arestas.")
    return G

def clusterize_graph_louvain(G, resolution=1.5):
    
    partition = community_louvain.best_partition(G, resolution=resolution, weight='weight')
    num_clusters = len(set(partition.values()))
    print(f"Clusterização completa. Número de clusters encontrados: {num_clusters}")
    return partition

def filter_small_clusters(partition, min_size=3):
    
    # Conta quantos nós há em cada cluster
    cluster_counts = {}
    for node, cluster in partition.items():
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    # Cria uma nova partição, atribuindo -1 para clusters pequenos
    filtered_partition = {}
    for node, cluster in partition.items():
        if cluster_counts[cluster] < min_size:
            filtered_partition[node] = -1  # Rótulo para ruído
        else:
            filtered_partition[node] = cluster
    return filtered_partition

def export_partition_to_csv(partition, output_file="clusters_resultado.csv"):
    
    rows = [{"Id": node, "modularity_class": cluster} for node, cluster in partition.items()]
    df_clusters = pd.DataFrame(rows, columns=["Id", "modularity_class"])
    df_clusters.to_csv(output_file, index=False)
    print(f"Arquivo CSV exportado com sucesso para {output_file}")

def main():
    
    file_path = "teste canonico\canonic_faiss_GRAFOS_10M(70S).csv"  
    df_edges = load_edges(file_path)
    if df_edges is None:
        return

    G = create_graph_from_edges(df_edges)
    
    
    partition = clusterize_graph_louvain(G, resolution=1.9)
    
    # Filtra os clusters pequenoss.
    partition_filtered = filter_small_clusters(partition, min_size=3)
    

    export_partition_to_csv(partition_filtered, output_file="teste canonico/clusters_resultado.csv")

main()