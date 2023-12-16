from typing import Literal

from params_proto import PrefixProto, Proto, Flag
from params_proto.partial import proto_partial

from watch_parser.few_shot import Featurizer
from hf.feedback import FeedbackModule


class FeatureArgs(PrefixProto):
    sp_range: Literal["short_term", "medium_term", "long_term"] = Proto(default="medium_term",
                                                                        help="The range of the Spotify API to use. Options: short_term, medium_term, long_term")
    num_artists: int = Proto(default=5,
                             help="Number of artists to use for feature generation, unless using cached features as demonstrations")
    songs_per_artist: int = Proto(default=10, help="Number of songs per artist to use for feature generation")
    verbose = Flag(default=True)
    csv_path: str = Proto(default="artists_data.csv", help="Path to the csv file containing example features")
    num_tracks: int = Proto(default=10, help="Number of tracks to produce")
    recommendation_max_retries: int = Proto(default=10, help="Maximum number of retries for recommendation generation")
    recommendation_sample_count: int = Proto(default=3,
                                             help="Number of features sample when generating recommendations")
    num_genres: int = Proto(default=5, help="Genre split for recommendation algorithm")
    use_cached_features = Flag(default=True,
                               help="Whether to use cached features as demonstrations. If true, will use the csv file specified by csv_path")


class RunnerArgs(PrefixProto):
    playlist_prefix: str = Proto(default="run-test-v1", help="Prefix to use for playlist generation")


class Runner:
    @proto_partial(RunnerArgs)
    def __init__(self, *, playlist_prefix: str):
        self.featurizer = Featurizer(**vars(FeatureArgs))
        self.feedback = FeedbackModule(None)

        self.feature_history = []
        self.genre_history = []

    def __matmul__(self, playlist_description):
        new_features, new_genres = self.featurizer(playlist_description)
        self.feature_history.append(new_features)
        self.genre_history.append(new_genres)

        return new_features, new_genres

    def __call__(self, playlist_description, activity_description=None):
        features, genres = self @ playlist_description
        self.featurizer.generate_playlist(features, genres, f"playlist_prefix/{len(self.feature_history)}")

        all_genres = self.featurizer.sp.recommendation_genre_seeds()["genres"]

        while True:
            feedback_prompt = str(input("Are you satisfied with your care? Enter feedback or Ctrl-C to exit: "))

            self.feedback.text_generator = self.featurizer.gen
            new_feature, new_genre = self.feedback(feedback_prompt, features, genres, all_genres, playlist_description)

            self.feature_history.append(new_feature)
            self.genre_history.append(new_genre)

            self.featurizer.generate_playlist(new_feature, new_genre, f"playlist_prefix/{len(self.feature_history)}")


if __name__ == '__main__':
    # initial_desc = "Energize Your Swim: A high-tempo mix of EDM, pop, and upbeat hits to keep your heart pumping and your strokes powerful. Perfect for moderate to high-intensity pool sessions."
    initial_desc = "Concentrated Study Beats: Enhance your focus with this playlist featuring classical and instrumental tracks. Ideal for deep concentration and productivity, these soothing, lyric-free melodies are perfect for any study session."
    runner = Runner()
    runner(initial_desc)
