# genre + target feature
# text -> LM -> change genres and features accordingly -> ...
# text -> LM -> "find me the coordinates of the feature that I should adjust, and up or down" -> lr * sign(LM)
from watch_parser.few_shot import Featurizer
import ast

class FeedbackModule:
    """
    Make alterations to the feature vector and genre list based on user feedback.

    """

    def __init__(self, text_generator, verbose=True):
        self.text_generator = text_generator
        self.verbose = verbose

    def feature_delta(self, human_feedback, previous_feature, playlist_description):
        """
        Ask LM to generate a new feature given the previous feature and the text.
        """
        feedback_prompt = (
            "Having already generated an initial feature vector {} based on the playlist description '{}', "
            "you now need to adjust this vector in light of the user's feedback: {}. "
            "Crucially, use the dataframe provided earlier, which contains example features and their corresponding genres, as a key reference in this process. "
            "Examine the typical values and trends within the genres in the dataframe to inform your adjustments. "
            "Your task is to interpret the user's feedback to modify the feature vector accurately, ensuring each updated value is consistent with the data patterns observed in the dataframe. "
            "Refine the vector so it more closely aligns with both the user's preferences and the musical characteristics of similar genres. "
            "Format your adjusted feature vector as a Python dictionary: "
            "{{'feature1': value1, 'feature2': value2, 'feature3': value3, ...}}. "
            "This adjustment should be data-driven, grounded in the specifics of the user's feedback and the genre characteristics in the dataframe. "
            "Present only the revised feature vector dictionary, without ANY additional commentary. "
            "This refinement is akin to customizing the song suggestions to better fit the user's musical tastes, using the dataframe as a guide for realistic and appropriate feature values."
        ).format(previous_feature, playlist_description, human_feedback)
        
        out_str = self.text_generator(feedback_prompt, verbose=self.verbose)
        try:
            new_feat = eval(out_str)
        except:
            dict_string = out_str[10:-4]
            new_feat = ast.literal_eval(dict_string)
            
        return new_feat

    def genre_delta(self, human_feedback, previous_genres, allowed_genres):
        """
        Similarly, ask LM to generate a new genre given the previous genres and the text.
        """
        feedback_prompt = (
            "Based on the user's feedback regarding their previous song recommendations, which is as follows: '{}', and your previous genre recommendations, which are as follows: '{}',"
            "determine the most suitable music genre(s) for their next listening experience. "
            "You have access to a range of genres: {}. Your task is to carefully analyze the user's feedback, "
            "then select and recommend genres from this list that closely match the user's music preferences as described in their feedback. "
            "Critically assess the defining characteristics of each genre in the context of the feedback. "
            "Present your genre recommendations in a Python list format. For example, if 'ambient', 'classical', and 'jazz' are the best matches, "
            "format your response as: ['ambient', 'classical', 'jazz']. "
            "Ensure that your selections are directly informed by the user's feedback, focusing on the specific aspects of the music they enjoyed or did not enjoy. "
            "Your genre recommendations should accurately reflect the musical preferences and style indicated in the user's feedback."
        ).format(human_feedback, previous_genres, ', '.join(allowed_genres))

        out = self.text_generator(feedback_prompt, verbose=self.verbose)
        out = eval(out)

        return out

    def __call__(self, human_feedback, previous_feature, previous_genres, allowed_genres, playlist_description):
        feature_delta = self.feature_delta(human_feedback, previous_feature, playlist_description)
        self.text_generator.reset_conversation()

        genre_delta = self.genre_delta(human_feedback, previous_genres, allowed_genres)
        self.text_generator.reset_conversation()

        return feature_delta, genre_delta


if __name__ == '__main__':
    from killport import kill_ports

    kill_ports(ports=[9090])
    desc = "Concentrated Study Beats: Enhance your focus with this playlist featuring instrumental and ambient tracks. Ideal for deep concentration and productivity, these soothing, lyric-free melodies are perfect for any study session."

    featurizer = Featurizer("medium_term", verbose=True, recommendation_sample_count=3, num_genres=3,
                            use_cached_features=True)

    features, genres = featurizer(desc)
    featurizer.generate_playlist(features, genres, "pre_feedback")

    feedback = FeedbackModule(featurizer.gen)
    allowed_genres = featurizer.sp.recommendation_genre_seeds()["genres"]

    example_feedback = "that was not a great selection, I want fast cool hype beast music"

    featurizer.generate_playlist(*feedback(example_feedback, features, genres, allowed_genres, desc), "post_feedback")
