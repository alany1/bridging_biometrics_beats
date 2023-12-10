import pickle


def embed_text(text):
    """
    Uses HuggingFace sentence-transformers/all-MiniLM-L6-v2 model to map sentences & paragraphs 
    to a 384 dimensional dense vector space for use as input to playlist generation model. 
    Returns vector space.
    """
    # Open saved pickle file 
    with open('../MINILM12.pkl', 'rb') as f:
        embedder = pickle.load(f)

    input_to_model = embedder.encode(text)
    return input_to_model




