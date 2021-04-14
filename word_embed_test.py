from gensim.models import Word2Vec

model = Word2Vec.load("word_embed.model")
out = model.wv.most_similar('meme', topn=20)
for w, v in out:
    print("{} : {}".format(w, v))

