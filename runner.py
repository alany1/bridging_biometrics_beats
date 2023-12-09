from typing import Literal

from params_proto import PrefixProto, Proto, Flag
from params_proto.partial import proto_partial

from watch_parser.alan_featurizer import Featurizer
from hf.feedback import FeedbackModule


class FeatureArgs(PrefixProto):
    sp_range: Literal["short_term", "medium_term", "long_term"] = Proto(default="medium_term",
                                                                        help="The range of the Spotify API to use. Options: short_term, medium_term, long_term")
    num_artists: int = Proto(default=5,
                             help="Number of artists to use for feature generation, unless using cached features as demonstrations")
    songs_per_artist: int = Proto(default=10, help="Number of songs per artist to use for feature generation")
    verbose = Flag(default=False)
    csv_path: str = Proto(default="artists_data.csv", help="Path to the csv file containing example features")
    num_tracks: int = Proto(default=20, help="Number of tracks to use for feature generation")
    recommendation_max_retries: int = Proto(default=10, help="Maximum number of retries for recommendation generation")
    recommendation_sample_count: int = Proto(default=5,
                                             help="Number of features sample when generating recommendations")
    num_genres: int = Proto(default=3, help="Genre split for recommendation algorithm")
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

    def __call__(self, playlist_description):
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
    initial_desc = "Concentrated Study Beats: Enhance your focus with this playlist featuring instrumental and ambient tracks. Ideal for deep concentration and productivity, these soothing, lyric-free melodies are perfect for any study session."
    runner = Runner()
    runner(initial_desc)

    # from killport import kill_ports
    # 
    # kill_ports(ports=[9090])
    # 
    # desc = "Concentrated Study Beats: Enhance your focus with this playlist featuring instrumental and ambient tracks. Ideal for deep concentration and productivity, these soothing, lyric-free melodies are perfect for any study session."
    # # desc = "Energize Your Swim: A high-tempo mix of EDM, pop, and upbeat hits to keep your heart pumping and your strokes powerful. Perfect for moderate to high-intensity pool sessions."
    # # desc = "Embrace the tranquility of the night with 'Midnight Stroll Serenade,' a playlist blending ambient sounds and soft, reflective melodies. Perfect for your nocturnal neighborhood walks, these tracks create a serene, contemplative atmosphere under the moonlit sky."
    # # desc = "Step into the calm of the night with 'Lunar Whisper: Midnight Walks,' a playlist featuring soothing Chinese songs. Ideal for a peaceful midnight stroll, these melodic tunes blend traditional instruments with modern sensibilities, creating a serene and reflective ambiance."
    # # desc = "'Morning Rise and Shine' - A vibrant playlist crafted for those crisp, early morning walks to the school bus stop. It's a blend of upbeat and energizing tracks to kick-start your day with positivity and motivation. Expect a mix of light-hearted pop, indie vibes, and inspiring tunes that mirror the freshness of a new day, perfect for a brisk walk under the morning sky."
    # # desc = "'Volley Vibes' - This dynamic playlist is designed to pump you up for your volleyball game warm-ups. It's a high-energy mix of motivating beats and powerful anthems that will get your adrenaline flowing. Expect a blend of intense electronic, upbeat pop, and driving rock tunes that are perfect for getting into the competitive spirit and preparing your body and mind for the game. The rhythm and energy of these tracks will keep you focused and ready to dominate the court."
    # featurizer = Featurizer("medium_term", verbose=True, recommendation_sample_count=3, num_genres=3,
    #                         use_cached_features=True)
    # # featurizer.sample_genres(ALL_GENRES)
    # features, genres = featurizer(desc)
    # featurizer.generate_playlist(features, genres, "pre_feedback")
    # 
    # feedback = FeedbackModule(featurizer.gen)
    # 
    # example_feedback = "that was a bit slow, I usually study with faster songs"
    # 
    # featurizer.gen.reset_conversation()
    # 
    # feedback_prompt = (
    #     "Having already generated an initial feature vector {} based on the playlist description '{}', "
    #     "you now need to adjust this vector in light of the user's feedback: {}. "
    #     "Crucially, use the dataframe provided earlier, which contains example features and their corresponding genres, as a key reference in this process. "
    #     "Examine the typical values and trends within the genres in the dataframe to inform your adjustments. "
    #     "Your task is to interpret the user's feedback to modify the feature vector accurately, ensuring each updated value is consistent with the data patterns observed in the dataframe. "
    #     "Refine the vector so it more closely aligns with both the user's preferences and the musical characteristics of similar genres. "
    #     "Format your adjusted feature vector as a Python dictionary: "
    #     "{{'feature1': value1, 'feature2': value2, 'feature3': value3, ...}}. "
    #     "This adjustment should be data-driven, grounded in the specifics of the user's feedback and the genre characteristics in the dataframe. "
    #     "Present only the revised feature vector dictionary, without additional commentary. "
    #     "This refinement is akin to customizing the song suggestions to better fit the user's musical tastes, using the dataframe as a guide for realistic and appropriate feature values."
    # ).format(features, desc, example_feedback)
    # 
    # x = featurizer.gen(feedback_prompt, verbose=True)
    # 
    # featurizer.gen.reset_conversation()
    # 
    # allowed_genres = featurizer.sp.recommendation_genre_seeds()["genres"]
    # 
    # prompt = (
    #     "Based on the user's feedback regarding their previous song recommendations, which is as follows: '{}', "
    #     "determine the most suitable music genre(s) for their next listening experience. "
    #     "You have access to a range of genres: {}. Your task is to carefully analyze the user's feedback, "
    #     "then select and recommend genres from this list that closely match the user's music preferences as described in their feedback. "
    #     "Critically assess the defining characteristics of each genre in the context of the feedback. "
    #     "Present your genre recommendations in a Python list format. For example, if 'ambient', 'classical', and 'jazz' are the best matches, "
    #     "format your response as: ['ambient', 'classical', 'jazz']. "
    #     "Ensure that your selections are directly informed by the user's feedback, focusing on the specific aspects of the music they enjoyed or did not enjoy. "
    #     "Your genre recommendations should accurately reflect the musical preferences and style indicated in the user's feedback."
    # ).format(example_feedback, ', '.join(allowed_genres))
    # 
    # y = featurizer.gen(prompt, verbose=True)
    # 
    # featurizer.generate_playlist(eval(x), eval(y), "post_feedback")
