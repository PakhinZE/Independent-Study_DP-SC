import unicodedata
import string
from pathlib import Path

DATA_SIZE = 100000

def check_word(word_check):
    all_letters = string.ascii_letters + " .,;'"
    # n_letters = len(all_letters)
    for character in word_check:
        if unicodedata.category(character) != "Mn" and character in all_letters:
            pass
        else:
            return False
    return True


def normalization_word(sentence, form="NFD"):
    sentence = unicodedata.normalize(form, sentence)  # type: ignore
    sentence = sentence.split()
    sentence = [word for word in sentence if check_word(word)]
    return " ".join(sentence)


def normalization_sentence_pair(sentence, sentence_noise, form="NFD"):
    sentence = unicodedata.normalize(form, sentence)  # type: ignore
    sentence_noise = unicodedata.normalize(form, sentence_noise)  # type: ignore
    sentence = sentence.split()
    sentence_noise = sentence_noise.split()
    sentence_return = list()
    sentence_noise_return = list()
    for word_check, word_noise_check in zip(sentence, sentence_noise):
        if check_word(word_check) and check_word(word_noise_check):
            sentence_return.append(word_check)
            sentence_noise_return.append(word_noise_check)
    return " ".join(sentence_return), " ".join(sentence_noise_return)


def readlines(readfile_path):
    sentences = open(readfile_path, encoding="utf-8").read().strip().split("\n")
    return [sentence for sentence in sentences]


def check_token_length(sentence, sentence_noise):
    if len(sentence.split()) != len(sentence_noise.split()):
        return False
    else:
        pass
    return True


if __name__ == "__main__":
    filename_list = [
        (
            "train.1blm",
            "train.1blm.noise.random",
        ),
        (
            "test.1blm",
            "test.1blm.noise.random",
        ),
    ]

    dataset_path = Path(__file__).parent.joinpath("dataset")
    dataset_norm_path = Path(__file__).parent.joinpath("normalization_dataset")

    for filename, filename_noise in filename_list:
        data_path = dataset_path.joinpath(filename)
        data_noise_path = dataset_path.joinpath(filename_noise)

        data_norm_path = dataset_norm_path.joinpath(f"normalization_{filename}")
        data_noise_norm_path = dataset_norm_path.joinpath(
            f"normalization_{filename_noise}"
        )

        file_save = open(data_norm_path, "w")
        file_noise_save = open(data_noise_norm_path, "w")

        document = readlines(data_path)
        document_noise = readlines(data_noise_path)

        print(data_path)
        print(data_noise_path)
        print(data_norm_path)
        print(data_noise_norm_path)
        
        size = DATA_SIZE

        for sentence, sentence_noise in zip(document, document_noise):
            sentence, sentence_noise = normalization_sentence_pair(
                sentence, sentence_noise
            )

            if not check_token_length(sentence, sentence_noise):
                print("Fail")
                break

            if sentence and sentence_noise:
                file_save.write(sentence)
                file_save.write("\n")
                file_noise_save.write(sentence_noise)
                file_noise_save.write("\n")
                size = size - 1
                print(size)
                if size == 0:
                    break
            

        file_save.close()
        file_noise_save.close()
