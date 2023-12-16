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

if __name__ == '__main__':
    # desc = "Concentrated Study Beats: Enhance your focus with this playlist featuring instrumental and ambient tracks. Ideal for deep concentration and productivity, these soothing, lyric-free melodies are perfect for any study session."
    desc = "Energize Your Swim: A high-tempo mix of EDM, pop, and upbeat hits to keep your heart pumping and your strokes powerful. Perfect for moderate to high-intensity pool sessions."
    out = embed_text(desc)


