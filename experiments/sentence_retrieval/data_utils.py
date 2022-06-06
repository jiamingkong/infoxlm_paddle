import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(HERE, "datasets")

__all__ = ["read_language_pairs", "get_language_pair_filenames", "language_mapping"]

language_mapping = {
    "ar": "ara",
    "bg": "bul",
    "de": "deu",
    "el": "ell",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "hi": "hin",
    "ru": "rus",
    "sw": "swh",
    "th": "tha",
    "tr": "tur",
    "ur": "urd",
    "vi": "vie",
    "zh": "cmn",
}


def get_language_pair_filenames(languageA, languageB):
    if languageA == "en":
        languageA = "eng"
        languageB = language_mapping[languageB]
    else:
        languageB = language_mapping[languageA]
        languageA = "eng"
    filenames = [
        os.path.join(DATASET_FOLDER, f"tatoeba.{languageB}-{languageA}.{languageA}"),
        os.path.join(DATASET_FOLDER, f"tatoeba.{languageB}-{languageA}.{languageB}"),
    ]
    return filenames


def read_language_pairs(filenames):
    with open(filenames[0], "r", encoding="utf-8") as f1, open(
        filenames[1], "r", encoding="utf-8"
    ) as f2:
        for line1, line2 in zip(f1, f2):
            yield line1.strip(), line2.strip()


if __name__ == "__main__":
    languageA, languageB = "en", "zh"
    filenames = get_language_pair_filenames(languageA, languageB)
    count = 5
    for line1, line2 in read_language_pairs(filenames):
        print(line1, line2)
        count -= 1
        if count == 0:
            break
