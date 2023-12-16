import pickle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
model2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#with open('/Users/michaelpeng/6.8611_final_project/MINILM12.pkl', 'wb') as file:
#    pickle.dump(model, file)
with open('/Users/michaelpeng/6.8611_final_project/MPNET.pkl', 'wb') as file:
    pickle.dump(model2, file)
