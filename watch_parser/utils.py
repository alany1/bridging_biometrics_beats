import numpy as np
import re
import ast


def to_np(features_dict):
    features_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                      'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    features = []
    for feature in features_names:
        features.append(features_dict[feature])
    return np.array(features)


def parse_outputs(output_string):
    # Regex pattern to extract dictionary
    dict_pattern = re.compile(r"\{[\s\S]*?\}")
    feature_vector_match = dict_pattern.search(output_string)
    feature_vector_str = feature_vector_match.group() if feature_vector_match else "{}"

    # Regex pattern to extract list
    list_pattern = re.compile(r"\[[\s\S]*?\]")
    genres_match = list_pattern.search(output_string)
    genres_str = genres_match.group() if genres_match else "[]"

    # Convert extracted strings to Python objects
    feature_vector = ast.literal_eval(feature_vector_str)
    genres = ast.literal_eval(genres_str)

    return feature_vector, genres
