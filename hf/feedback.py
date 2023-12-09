# genre + target feature
# text -> LM -> change genres and features accordingly -> ...
# text -> LM -> "find me the coordinates of the feature that I should adjust, and up or down" -> lr * sign(LM)

class FeedbackModule:
    def __init__(self, text_generator):
        self.text_generator = text_generator

    def feature_delta(self, text, previous_feature):
        """
        Ask LM to generate a new feature given the previous feature and the text.
        """
        prompt = f"Given the initial feature vector {previous_feature} and the user's feedback {text} on the suggested song, please make appropriate adjustments to the feature vector. Consider the specific dimensions of the feature vector and make changes that align with the user's preferences. Imagine this process as taking a step in the direction that improves the song suggestions based on the provided feedback."

        new_feat = eval(self.text_generator(prompt, verbose=self.verbose))
        return new_feat

    def genre_delta(self, text, previous_genres, allowed_genres):
        """
        Similarly, ask LM to generate a new genre given the previous genres and the text.
        """
        prompt = (
            "Based on the user's feedback regarding their previous song recommendations, which is as follows: '{}', "
            "determine the most suitable music genre(s) for their next listening experience. "
            "You have access to a range of genres: {}. Your task is to carefully analyze the user's feedback, "
            "then select and recommend genres from this list that closely match the user's music preferences as described in their feedback. "
            "Critically assess the defining characteristics of each genre in the context of the feedback. "
            "Present your genre recommendations in a Python list format. For example, if 'ambient', 'classical', and 'jazz' are the best matches, "
            "format your response as: ['ambient', 'classical', 'jazz']. "
            "Ensure that your selections are directly informed by the user's feedback, focusing on the specific aspects of the music they enjoyed or did not enjoy. "
            "Your genre recommendations should accurately reflect the musical preferences and style indicated in the user's feedback."
        ).format(text, ', '.join(allowed_genres))

        out = eval(self.text_generator(prompt, verbose=self.verbose))

        # postprocess

        return out

    def __call__(self, text, previous_feature, previous_genres):
        feature_delta = self.feature_delta(text, previous_feature)
        genre_delta = self.genre_delta(text, previous_genres)
        return feature_delta, genre_delta


if __name__ == '__main__':
    pass
