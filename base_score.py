from pathlib import Path
import Model

from torchmetrics.text import CharErrorRate

DATA_SIZE = 100000

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
        data_norm_path = dataset_norm_path.joinpath(f"normalization_{filename}")
        data_noise_norm_path = dataset_norm_path.joinpath(
            f"normalization_{filename_noise}"
        )

        document = Model.readlines(data_norm_path)
        document_noise = Model.readlines(data_noise_norm_path)

        print(data_noise_norm_path)
        print(data_norm_path)

        char_error_rate = CharErrorRate()
        score = char_error_rate(document_noise[0:DATA_SIZE], document[0:DATA_SIZE])
        print(f"Base Score For Spell Correction: {score}")