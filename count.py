from pathlib import Path


def readlines(filename):
    sentences = open(filename, encoding="utf-8").read().strip().split("\n")
    return [sentence for sentence in sentences]


if __name__ == "__main__":
    dataset_path = Path(__file__).parent.joinpath("dataset")
    dataset_norm_path = Path(__file__).parent.joinpath("normalization_dataset")

    filename_list = {
        "Train": dataset_path.joinpath("train.1blm"),
        "Train Norm": dataset_norm_path.joinpath("normalization_train.1blm"),
        "Train Noisy": dataset_path.joinpath("train.1blm.noise.random"),
        "Train Noisy Norm": dataset_norm_path.joinpath(
            "normalization_train.1blm.noise.random"
        ),
        "Test": dataset_path.joinpath("test.1blm"),
        "Test Norm": dataset_norm_path.joinpath("normalization_test.1blm"),
        "Test Noisy": dataset_path.joinpath("test.1blm.noise.random"),
        "Test Noisy Norm": dataset_norm_path.joinpath(
            "normalization_test.1blm.noise.random"
        ),
    }

    for name, file_path in filename_list.items():
        document = readlines(file_path)
        print(f"Number of {name} Sentence: {len(document)}")
        token_count = 0
        for sentence in document:
            token_count = token_count + len(sentence.split())
        print(f"Number of {name} Token: {token_count}")

''''
Number of Train Sentence: 1365669
Number of Train Token: 30431058
Number of Train Norm Sentence: 1365633
Number of Train Norm Token: 28826549
Number of Train Noisy Sentence: 1365669
Number of Train Noisy Token: 30431058
Number of Train Noisy Norm Sentence: 1365633
Number of Train Noisy Norm Token: 28826549
Number of Test Sentence: 273134
Number of Test Token: 6085104
Number of Test Norm Sentence: 273129
Number of Test Norm Token: 5763498
Number of Test Noisy Sentence: 273134
Number of Test Noisy Token: 6085104
Number of Test Noisy Norm Sentence: 273129
Number of Test Noisy Norm Token: 5763498
'''
