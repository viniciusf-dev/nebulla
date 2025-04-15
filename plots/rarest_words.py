import os
import json
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))

json_path = os.path.join(script_dir, '..', 'nebula_model.json')

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

vocab_dict = data["vocabulary"]["word_to_index"]
idf_list = data["vocabulary"]["idf_values"]

index_to_word = {idx: word for word, idx in vocab_dict.items()}

words_idf = [(index_to_word[i], idf_list[i]) for i in range(len(idf_list))]

words_idf.sort(key=lambda x: x[1], reverse=True)

top_20 = words_idf[:20]

words, idfs = zip(*top_20)

plt.bar(range(len(words)), idfs)
plt.title("Top 20 Words by Highest IDF")
plt.xticks(range(len(words)), words, rotation=90)
plt.xlabel("Word")
plt.ylabel("IDF")
plt.tight_layout()
plt.show()
