import pandas as pd

def criar_matriz_usuario_filme(caminho_arquivo, limite_nota=3, caminho_saida='base bruta/100k/matriz_usuario_filme.csv'):    
    try:
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

    # aplica o threshold: notas < limite_nota (3) são definidas como not a number
    dados['rating'] = dados['rating'].apply(lambda x: x if x >= limite_nota else pd.NA)
    print(f"Aplicado threshold de notas menores que {limite_nota} para serem descartadas.")

    # Criar a matriz usuário x filme
    matriz = dados.pivot_table(index='userId', columns='movieId', values='rating')
    print("Matriz usuário x filme criada com sucesso.")

    # Salva matriz em um arquivo CSV
    try:
        matriz.to_csv(caminho_saida)
        print(f"Matriz salva com sucesso em {caminho_saida}.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo: {e}")

    return matriz


#parte de execução
caminho_arquivo = 'base bruta/1M/ratings.csv'  
matriz_usuario_filme = criar_matriz_usuario_filme(
    caminho_arquivo=caminho_arquivo,
    limite_nota=3,
    caminho_saida='base organizada/1M/matriz_usuario_filme_1M.csv'
)

print(matriz_usuario_filme.head()) #função so pra ver as linhas iniciais e ver se deu certo
