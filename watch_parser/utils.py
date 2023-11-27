import numpy as np
import ast

def to_np(features_dict):
    features_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                      'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    features = []
    for feature in features_names:
        features.append(features_dict[feature])
    return np.array(features)

def parse_outputs(output_string):
    # Split the output into two parts
    feature_vector_part, genres_part = output_string.strip().split('\n\n', 1)

    # Remove Markdown code block syntax
    feature_vector_str = feature_vector_part.replace('```python\n', '').replace('\n```', '')
    genres_str = genres_part.replace('```python\n', '').replace('\n```', '')

    # Convert strings to Python objects
    feature_vector = ast.literal_eval(feature_vector_str)
    genres = ast.literal_eval(genres_str)

    return feature_vector, genres
