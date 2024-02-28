"""
Define the dataset used in the experimentation
"""
import pandas as pd


class IMDBDataset:
    def __init__(self, imdb_dataset_path: str):
        """
        Initialize the IMDB dataset
        :param imdb_dataset_path: the path of the imdb dataset
        """
        # Read in the dataset
        self.imdb = pd.read_csv(imdb_dataset_path)

        # Used for iterator
        self.num = 0

        # Read in the sentences and labels
        self.texts, self.labels = [], []
        for index, row in self.imdb.iterrows():
            self.texts.append(row['review'])
            self.labels.append(0 if row['sentiment'] == 'negative' else 1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num >= len(self.texts):
            raise StopIteration
        item = self.texts[self.num], self.labels[self.num]
        self.num += 1
        return item