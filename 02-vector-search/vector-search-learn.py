"""
Vector search is a technique that uses the cosine similarity rule to find similar items in a high-dimensional space. 
In this example, we will use the SentenceTransformer library to encode sentences into vectors and perform vector search using dot product similarity.

Cosine similarity measures the angle between two vectors, ignoring their length:

1.0 = same direction (similar)
0.0 = perpendicular (unrelated)
-1.0 = opposite direction (opposite meaning)
Formally, if theta is the angle between two vectors, cosine similarity is cos(theta):

cos(0) = 1 - vectors point in the same direction
cos(90) = 0 - vectors are perpendicular
cos(180) = -1 - vectors point in opposite directions
Because our vectors are normalized, the dot product gives us cosine similarity directly. This is why we can use v1.dot(dv) to compare texts.

"""
# from sentence_transformers import SentenceTransformer

"""# Test the vector search with some example sentences"""
# model = SentenceTransformer("all-MiniLM-L6-v2")

# q1 = "Can I still join the course after the start date?"
# v1 = model.encode(q1)

# d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
# dv = model.encode(d)

# print(v1.dot(dv))

# q2 = "How to install Docker on Windows?"
# v2 = model.encode(q2)

# print(v2.dot(dv))



""" Loading the data: Let's get data from the ingest.py, and build. """
from ingest import load_faq_data

documents = load_faq_data()
# print(type(documents))
# print(documents[2])
# print(f"Loaded {len(documents)} documents.")



"""Generating embeddings """
texts = []
for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)


# print(f"Type of text is {type(texts)}")
# print(f"Length of text is {len(texts)}")
# print(f"First 3: {texts[0:3]}")    

"""Next we chunk the dataset into batches of 50 and encode each batch:"""

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

# We set the number of texts we want to process at once to 50
batch_size = 50
model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = []

# Note: We end up with 1208 vectors. On a GPU this is fast. Most of us run on Codespaces without a GPU, so it takes a bit, but it's a one-off.

# 1. range(0, len(texts), batch_size) generates starting indices: 0, 50, 100, 150, ...
# 2. tqdm(...) wraps this loop to display a visual progress bar in the terminal
for i in tqdm(range(0, len(texts), batch_size)):
    
    # 3. Slice the list of texts from index 'i' up to 'i + 50' to get a batch of 50 items
    batch = texts[i:i + batch_size]
    
    # 4. Encode the entire batch of 50 texts simultaneously into vectors
    batch_vectors = model.encode(batch)
    
    # 5. Append these 50 new vectors to our master list
    vectors.extend(batch_vectors)

# len(vectors)
""" 
We turn them into a 2-dimensional array (matrix) where
- rows are documents (vectors)
- columns are dimensions of the vectors
"""

import numpy as np
X = np.array(vectors)
# print(X.shape)
# (1401, 384)

"""
Let's perform vector search

--- HOW IT WORKS UNDER THE HOOD
"""

query = "Can I still join the course after the start date?"
v_query = model.encode(query)

scores = X.dot(v_query)
# print(len(scores))
# print(scores.max())

# The highest score is the most similar document:
idx = np.argmax(scores)
print(idx, scores[idx])

# Let's check the result:
print(documents[idx])

""" 
print(scores.argmax())
This returns the index of the most similar document.
"""

top5 = np.argsort(-scores)[:5]
for idx in top5:
    print(scores[idx])
    print(documents[idx])
    print()

"""
This is vector search in its simplest form. We embed the query, compute dot products against all documents, and return the highest-scoring ones.

We return 5 and not the single best for a reason. The answer to a question can be spread across several documents. One holds part of it, another fills in the rest. Sometimes the top result isn't the right one but the second is. We send all 5 to the LLM and let it combine them.

The number 5 is a starting point, picked on gut feeling. Later, when we evaluate search quality, we can test whether 3 or 10 works better for our data.

Doing this by hand with numpy is fine for a small dataset. A larger one needs a library that also handles filtering and ranking. That's what we turn to next.

"""