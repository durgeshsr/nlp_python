from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import gensim
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

#download is one time activity, hence disabled for now
#nltk.download()
'''
def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()
'''


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
      model2.wv.similarity('little', 'crocodile'))

print("Similarity between 'alice' " +
      "and 'beginning' - Skip Gram : ",
      model2.wv.similarity('alice', 'beginning'))

my_dict = dict({})
for idx, key in enumerate(model2.wv.vocab):
    my_dict[key] = model2.wv[key]

for key, value in my_dict.items():
    pass
    #print(key, value)


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

#display_closestwords_tsnescatterplot(model2, 'alice')



