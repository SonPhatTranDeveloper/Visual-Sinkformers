"""
This file is a utility to convert the IMDB dataset into a corpus file
"""
import pandas as pd

if __name__ == "__main__":
    # Data file
    INPUT_FILE_PATH = "../data/imdb_dataset.csv"
    OUTPUT_FILE_PATH = "../data/imdb_corpus.txt"

    # Open input file
    imdb = pd.read_csv(INPUT_FILE_PATH)

    # Open new file and write the content
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as file:
        for sentence in imdb["review"]:
            # Write the sentence into the output file
            file.write(sentence + "\n")
