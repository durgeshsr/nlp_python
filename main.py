from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

#download is one time activity, hence disabled for now
#nltk.download()

sample = open("D:\\durgesh.ramani\\My Downloads\\gutenburg_trainingdata.txt", encoding="utf8")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, size=100,
                                window=10, sg=1)

# Print results
print("Similarity between 'next' " +
      "and 'question' - Skip Gram : ",
      model2.wv.similarity('next', 'question'))

print("Similarity between 'alice' " +
      "and 'beginning' - Skip Gram : ",
      model2.wv.similarity('alice', 'beginning'))

# below code is used to show scatterplot in 2d
vocab = list(model2.wv.vocab)
X = model2[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()



