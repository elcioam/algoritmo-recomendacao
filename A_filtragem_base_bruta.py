import pandas as pd

def criar_matriz_usuario_filme(caminho_arquivo, caminho_saida='base bruta/100k/matriz_usuario_filme.csv'):    
    try:
        # Lê somente as colunas necessárias
        dados = pd.read_csv(caminho_arquivo, usecols=['userId', 'movieId', 'rating'])
        print("Dados importados com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo {caminho_arquivo} não foi encontrado.")
        return
    except pd.errors.EmptyDataError:
        print("Erro: O arquivo está vazio.")
        return
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return

    # Aqui, todas as notas são consideradas, sem aplicar threshold.
    print("Todas as notas foram consideradas, sem filtragem.")

    # Criar a matriz usuário x filme, mantendo os vazios onde não há avaliação
    matriz = dados.pivot_table(index='userId', columns='movieId', values='rating')
    print("Matriz usuário x filme criada com sucesso.")

    # Salva a matriz em um arquivo CSV
    try:
        matriz.to_csv(caminho_saida)
        print(f"Matriz salva com sucesso em {caminho_saida}.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo: {e}")

    return matriz


caminho_arquivo = 'bases MovieLens/10M/ratings.csv'  
matriz_usuario_filme = criar_matriz_usuario_filme(
    caminho_arquivo=caminho_arquivo,
    caminho_saida='bases utilizadas/matriz_usuario_filme_10M.csv'
)

