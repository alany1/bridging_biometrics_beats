from textgenerator import TextGenerator
from embedding_to_feature import generate_params
from embedding import embed_text

dataset = "../example_data/swim_merged.csv"
prompt = "This was a swimming workout. What genre of music would be good to listen to? Using all of the data provided, come to a conclusion. Summarize it as a playlist description, and only return the description."
gen = TextGenerator("../example_data/swim_merged.csv")
lm_output = gen(prompt)
embedding = embed_text(lm_output)
features = generate_params(embedding)
print(features)