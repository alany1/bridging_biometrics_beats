import numpy as np
import pandas as pd
from spotipy import SpotifyOAuth

import watch_parser
from watch_parser.textgenerator import TextGenerator
from typing import Literal
import spotipy


class Featurizer:
    """
    Extract a spotify feature vector from LM output by providing demonstration data
    
    1. Pulls the users top artists in specified range
    2. Extract audio features for the top songs of each artist
    3. Average features for each artist and get their genres
    4. Create a dataframe with the features and genres
    5. Save the dataframe to csv
    
    To get a feature vector from this:
    1. Load the csv to the LM
    2. Feed the LM a prompt with the desired description
    3. The LM will output genres and a feature vector
    4. Features and genres sent to SP
    """
    features_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                      'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

    output_features_names = ['target_danceability', 'target_energy', 'target_key', 'target_loudness', 'target_mode',
                             'target_speechiness', 'target_acousticness', 'target_instrumentalness', 'target_liveness',
                             'target_valence', 'target_tempo', 'target_time_signature']

    def __init__(self, sp_range: Literal["short_term", "medium_term", "long_term"], num_artists=5, songs_per_artist=50,
                 verbose=False, csv_path="artists_data.csv", num_tracks=20):
        self.sp_range = sp_range
        self.num_artists = num_artists
        self.songs_per_artist = songs_per_artist

        self.verbose = verbose
        self.csv_path = csv_path

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-top-read"))

        self.num_tracks = num_tracks

        self.df_features = None
        self.gen = None
        self.prompt = None

        self.text_generator = None

    def get_top_artists(self):
        artist_genres = []
        top_artists = []

        results = self.sp.current_user_top_artists(time_range=self.sp_range, limit=self.num_artists)
        for i, item in enumerate(results['items']):
            top_artists.append(item['name'])
            artist_genres.append(item['genres'])

        return top_artists, artist_genres

    def get_artist_features(self, top_artists):
        artist_features = []
        for artist_name in top_artists:
            artist_tracks = self.sp.search(q=artist_name, limit=self.songs_per_artist)
            tids = []
            for i, t in enumerate(artist_tracks['tracks']['items']):
                if self.verbose:
                    print(' ', i, t['name'])
                tids.append(t['uri'])
            features = self.sp.audio_features(tids)

            features_np = []
            for feature in features:
                if feature is None:
                    if self.verbose:
                        print("Skipping song from", artist_name)
                    continue
                features_np.append(watch_parser.to_np(feature))

            artist_features.append(np.mean(features_np, axis=0))

        return artist_features

    def create_dataframe(self, artist_features, artist_genres):
        self.df_features = pd.DataFrame(np.array(artist_features), columns=self.features_names)
        self.df_features['Genre'] = artist_genres
        self.df_features.to_csv(self.csv_path, index=False)

    def get_target_features(self, lm_description):
        self.gen = TextGenerator(self.csv_path)
        feat_prompt = (
            f"You generated the following playlist description: '{lm_description}'. "
            f"Based on this description, generate a feature vector with components: "
            f"{Featurizer.output_features_names}. Format your response as a Python dictionary. "
            f"For example, return the feature vector like this: "
            f"{{'feature1': value1, 'feature2': value2, 'feature3': value3}}. "
            f"Return only the dictionary without any additional commentary."
        )

        feat_output = self.gen(feat_prompt, verbose=self.verbose)

        all_genres = self.sp.recommendation_genre_seeds()["genres"]
        genre_prompt = (
            f"Based on the list of genres: {all_genres}, identify a few genres that match the description. "
            f"Format your response as a Python list. For example, return the genres like this: "
            f"['genre1', 'genre2', 'genre3']."
        )

        # all outputs are stored in this variable
        genre_output = self.gen(genre_prompt, verbose=self.verbose)

        features, genres = watch_parser.parse_outputs(genre_output)

        return features, genres

    def _get_recommendations(self, target_features, target_genres):
        results = self.sp.recommendations(seed_genres=target_genres, limit=self.num_tracks, **target_features)
        tracks_to_add = []
        for track in results['tracks']:
            if self.verbose:
                print(track['name'], '-', track['artists'][0]['name'])
            tracks_to_add.append(track['uri'])
        return tracks_to_add

    def _make_playlist(self, tracks_to_add, playlist_name):
        self.sp.user_playlist_create(self.sp.me()['id'], playlist_name)

        playlist = self.sp.user_playlists(self.sp.me()['id'], limit=1, offset=0)
        playlist = playlist['items'][0]['id']

        self.sp.playlist_add_items(playlist_id=playlist['id'], items=tracks_to_add)

    def __call__(self, lm_description):
        top_artists, artist_genres = self.get_top_artists()
        artist_features = self.get_artist_features(top_artists)

        self.create_dataframe(artist_features, artist_genres)

        target_features, target_genres = (
            {'target_danceability': 0.7, 'target_energy': 0.8, 'target_key': 5, 'target_loudness': -5.0,
             'target_mode': 1,
             'target_speechiness': 0.1, 'target_acousticness': 0.2, 'target_instrumentalness': 0.05,
             'target_liveness': 0.2,
             'target_valence': 0.6, 'target_tempo': 120.0, 'target_time_signature': 4},
            ['edm', 'pop', 'dance', 'work-out'])

        # target_features, target_genres = self.get_target_features(lm_description)

        return target_features, target_genres

    def generate_playlist(self, target_features, target_genres, playlist_name):
        tracks_to_add = self._get_recommendations(target_features, target_genres)
        self._make_playlist(tracks_to_add, playlist_name)

        print(f"Playlist {playlist_name} created!")


if __name__ == '__main__':
    featurizer = Featurizer("short_term", verbose=True)
    featurizer(
        "Energize Your Swim: A high-tempo mix of EDM, pop, and upbeat hits to keep your heart pumping and your strokes powerful. Perfect for moderate to high-intensity pool sessions.")