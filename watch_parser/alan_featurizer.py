import numpy as np
import pandas as pd
import requests.exceptions
from spotipy import SpotifyOAuth

import watch_parser
from watch_parser.textgenerator import TextGenerator
from typing import Literal
import spotipy
import requests
import random

"""
I feel that it should be quite sensitive to the initial artist pool (i.e. the top artists of the user). But somehow it still did quite well
However, we should diversify (maybe hand-select) the initial pool to help it generalize better 
"""


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

    def __init__(self, sp_range: Literal["short_term", "medium_term", "long_term"], num_artists=5, songs_per_artist=10,
                 verbose=False, csv_path="artists_data.csv", num_tracks=20, recommendation_max_retries=10,
                 recommendation_sample_count=6, num_genres=3):
        self.sp_range = sp_range
        self.num_artists = num_artists
        self.songs_per_artist = songs_per_artist

        self.verbose = verbose
        self.csv_path = csv_path

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-top-read playlist-modify-public"))

        self.num_tracks = num_tracks

        self.df_features = None
        self.gen = None
        self.prompt = None

        self.text_generator = None

        self.num_genres = num_genres

        self.recommendation_max_retries = recommendation_max_retries
        self.recommendation_sample_count = recommendation_sample_count

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
        try:
            # Since we can only have 5 seeds, we will sample 3 genres and two artists
            my_top_artists = None
            if self.num_genres < 5:
                num_genres = min(self.num_genres, len(target_genres), 5)
                target_genres = random.sample(target_genres, num_genres)
                get = self.sp.current_user_top_artists(time_range=self.sp_range, limit=4 * (5 - num_genres))
                my_top_artists = random.sample([artist['id'] for artist in get['items']], 5 - num_genres)
            results = self.sp.recommendations(seed_genres=target_genres[:5], seed_artists=my_top_artists,
                                              limit=self.num_tracks, **target_features)
        except (requests.exceptions.HTTPError, spotipy.exceptions.SpotifyException):
            while self.recommendation_max_retries > 0:
                print("Features were too precise. Retrying recommendation request...")
                self.recommendation_max_retries -= 1
                target_keys = random.sample(list(target_features.keys()), self.recommendation_sample_count)
                new_features = {key: target_features[key] for key in target_keys}
                try:
                    results = self.sp.recommendations(seed_genres=target_genres[:5], seed_artists=my_top_artists,
                                                      limit=self.num_tracks, **new_features)
                except (requests.exceptions.HTTPError, spotipy.exceptions.SpotifyException):
                    continue
                print(f"Recommendation request succeeded with keys {target_keys}!")
                break
            else:
                raise RuntimeError("Failed to get recommendations")
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

        self.sp.playlist_add_items(playlist_id=playlist, items=tracks_to_add)

    def __call__(self, lm_description):
        top_artists, artist_genres = self.get_top_artists()
        artist_features = self.get_artist_features(top_artists)

        self.create_dataframe(artist_features, artist_genres)

        target_features, target_genres = self.get_target_features(lm_description)

        return target_features, target_genres

    def generate_playlist(self, target_features, target_genres, playlist_name):
        tracks_to_add = self._get_recommendations(target_features, target_genres)
        self._make_playlist(tracks_to_add, playlist_name)

        print(f"Playlist {playlist_name} created!")


if __name__ == '__main__':
    # desc = "Concentrated Study Beats: Enhance your focus with this playlist featuring instrumental and ambient tracks. Ideal for deep concentration and productivity, these soothing, lyric-free melodies are perfect for any study session."
    # desc = "Energize Your Swim: A high-tempo mix of EDM, pop, and upbeat hits to keep your heart pumping and your strokes powerful. Perfect for moderate to high-intensity pool sessions."
    desc = "Embrace the tranquility of the night with 'Midnight Stroll Serenade,' a playlist blending ambient sounds and soft, reflective melodies. Perfect for your nocturnal neighborhood walks, these tracks create a serene, contemplative atmosphere under the moonlit sky."
    # desc = "Step into the calm of the night with 'Lunar Whisper: Midnight Walks,' a playlist featuring soothing Chinese songs. Ideal for a peaceful midnight stroll, these melodic tunes blend traditional instruments with modern sensibilities, creating a serene and reflective ambiance."
    # desc = "'Morning Rise and Shine' - A vibrant playlist crafted for those crisp, early morning walks to the school bus stop. It's a blend of upbeat and energizing tracks to kick-start your day with positivity and motivation. Expect a mix of light-hearted pop, indie vibes, and inspiring tunes that mirror the freshness of a new day, perfect for a brisk walk under the morning sky."
    # desc = "'Volley Vibes' - This dynamic playlist is designed to pump you up for your volleyball game warm-ups. It's a high-energy mix of motivating beats and powerful anthems that will get your adrenaline flowing. Expect a blend of intense electronic, upbeat pop, and driving rock tunes that are perfect for getting into the competitive spirit and preparing your body and mind for the game. The rhythm and energy of these tracks will keep you focused and ready to dominate the court."
    featurizer = Featurizer("medium_term", verbose=True, recommendation_sample_count=3, num_genres=3)
    features, genres = featurizer(desc)
    featurizer.generate_playlist(features, genres, "alan_test")
