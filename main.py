def main():
    """
    Vector search is a technique that uses the cosine similarity rule to find similar items in a high-dimensional space. 
    In this example, we will use the SentenceTransformer library to encode sentences into vectors and perform vector search using dot product similarity.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    q1 = "Can I still join the course after the start date?"
    v1 = model.encode(q1)

    d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
    dv = model.encode(d)

    print(v1.dot(dv))

    q2 = "How to install Docker on Windows?"
    v2 = model.encode(q2)

    print(v2.dot(dv))

    print("Hello from llm-zoomcamp!")


if __name__ == "__main__":
    main()
