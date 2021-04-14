import sys
import re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec

# import nltk
# nltk.download()
# exit()

STOP_WORDS = set(stopwords.words('english'))

def parse_sentence_words(input_file_names):
    """Returns a list of a list of words. Each sublist is a sentence."""
    sentence_words = []
    for file_name in input_file_names:
        for line in open(file_name, 'r'):
            sent_words = [[w for w in re.findall(r'\b(\w+)\b', s) if w not in STOP_WORDS] for s in sent_tokenize(line.strip().lower())]
            if len(sent_words) > 1:
                sentence_words += filter(lambda sw: len(sw) > 1, sent_words)
    return sentence_words


input_file_names = [
    "doriangray.txt"
    ]

print("Finding the words...")
THE_WORDS = parse_sentence_words(input_file_names)

print("Training the model...")
model = Word2Vec(sentences=THE_WORDS, vector_size=128, window=7, min_count=5, workers=4)
model.save("word_embed2.model")

print("Sample word : word")
out = model.wv.most_similar('word', topn=20)

for w, v in out:
    print("{} : {}".format(w, v))

