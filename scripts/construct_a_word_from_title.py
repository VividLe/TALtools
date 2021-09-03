from nltk.corpus import wordnet
import random


def is_word(word):
    result = wordnet.synsets(word)
    if len(result) != 0:
        return True
    else:
        return False


title = 'Background Click Supervision'
patience = 30000

words = title.split(' ')

legal_spelling = list()
for _ in range(patience):
    spelling = []
    for word in words:
        char = random.sample(word, 1)[0]
        spelling.append(char)

    spelling = ''.join(spelling)
    if is_word(spelling):
        if spelling not in legal_spelling:
            legal_spelling.append(spelling)

for s in legal_spelling:
    print(s)

