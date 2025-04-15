import os
import json
import matplotlib.pyplot as plt

# Obtém o diretório onde o script está localizado
script_dir = os.path.dirname(os.path.realpath(__file__))

# Constrói o caminho para o arquivo JSON subindo um nível (root)
json_path = os.path.join(script_dir, '..', 'nebula_model.json')

# 1) Carregar o arquivo JSON
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2) Extrair dados relevantes
vocab_dict = data["vocabulary"]["word_to_index"]
idf_list = data["vocabulary"]["idf_values"]

# 3) Inverter o dicionário para {índice -> palavra} para acessarmos pelas posições
index_to_word = {idx: word for word, idx in vocab_dict.items()}

# Criar uma lista de (palavra, idf)
words_idf = [(index_to_word[i], idf_list[i]) for i in range(len(idf_list))]

# Ordenar por IDF (maior -> menor)
words_idf.sort(key=lambda x: x[1], reverse=True)

# Selecionar as top 20 palavras com maior IDF
top_20 = words_idf[:20]

# Separar as colunas para plotagem
words, idfs = zip(*top_20)

# Criar o gráfico de barras
plt.bar(range(len(words)), idfs)
plt.title("Top 20 Words by Highest IDF")
plt.xticks(range(len(words)), words, rotation=90)
plt.xlabel("Word")
plt.ylabel("IDF")
plt.tight_layout()
plt.show()
