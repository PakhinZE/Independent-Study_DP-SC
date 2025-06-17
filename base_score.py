from pathlib import Path
import model

import torch
from torchmetrics.text import CharErrorRate

if __name__ == "__main__":
    DATA_SIZE = None  # None
    DEVICE = "cuda"
    
    torch.set_default_device(DEVICE)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

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

        document = model.readlines(data_norm_path)
        document_noise = model.readlines(data_noise_norm_path)

        print(data_norm_path)
        print(data_noise_norm_path)

        char_error_rate = CharErrorRate()
        if DATA_SIZE:
            score = char_error_rate(document_noise[0:DATA_SIZE], document[0:DATA_SIZE])
        else:
            score = char_error_rate(document_noise, document)
        print(f"Base Score for Spell Correction (Char Error Rate): {score}")
