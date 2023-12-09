import numpy as np
import re
import ast

ALL_GENRES = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal', 'bluegrass',
              'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop', 'chicago-house', 'children', 'chill',
              'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house',
              'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic',
              'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove',
              'grunge', 'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays',
              'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance',
              'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal',
              'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode',
              'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 'progressive-house',
              'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day', 'reggae', 'reggaeton', 'road-trip', 'rock',
              'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes',
              'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks', 'spanish', 'study', 'summer',
              'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'trip-hop', 'turkish', 'work-out', 'world-music']


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
