import pandas as pd
from functools import reduce


class Dataparser:
    """
    Responsible for collecting and merging data from the same activity.
    """

    def __init__(self, *paths, merge_keys=["Date/Time"], drop_keys=["Source"], join_method="inner"):
        self.data_paths = paths
        self.merge_keys = merge_keys
        self.drop_keys = drop_keys
        self.join_method = join_method

    def __matmul__(self, destination_path):
        """
        Generate a single dataframe from the data_paths.
        """
        dfs = [pd.read_csv(path).drop(self.drop_keys, axis=1) for path in self.data_paths]
        merged_df = reduce(lambda left, right: pd.merge(left, right, on=self.merge_keys, how=self.join_method), dfs)
        
        merged_df.sort_values(by=self.merge_keys, inplace=True)
        merged_df.to_csv(destination_path, index=False)


if __name__ == '__main__':
    paths = ["../example_data/swim_hr.csv", "../example_data/swim_calories.csv", "../example_data/swim_dist.csv",
             "../example_data/swim_stroke-count.csv"]
    parser = Dataparser(*paths)
    df = parser @ "../example_data/swim_merged.csv"
