import pickle
from pathlib import Path

import torch
import torch.utils.data

import Model


if __name__ == "__main__":
    PATH = Path(__file__).parent
    DEVICE = "cuda"

    # Data Size
    DATA_SIZE = 100000
    # Loader
    LOADER_WORKER = 0  # Default = 0
    BATCH_SIZE = 32
    # NN
    HIDDEN_SIZE = 512
    NUM_LAYER = 2  #
    DROPOUT = 0.4
    # Train
    NUM_EPOCHS = 45
    PATIENCE = 3
    LABEL_SMOOTH = 0.1

    # Debug
    DEBUG = False
    DEBUG_EPOCH = False

    torch.set_default_device(DEVICE)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    voc_path = PATH.joinpath("voc").absolute()
    with open(voc_path, "rb") as voc_file:
        voc = pickle.load(voc_file)

    WORD_SIZE = len(voc["token2idx"])
    CHAR_SIZE = len(voc["chartoken2idx"])
    SEMI_CHAR_VEC_SIZE = CHAR_SIZE * 3

    noise_train_dataset_path = PATH.joinpath("normalization_dataset/").joinpath(
        "normalization_train.1blm.noise.random"
    )
    label_train_dataset_path = PATH.joinpath("normalization_dataset/").joinpath(
        "normalization_train.1blm"
    )

    train_dataset = Model.spell_correction_dataset(
        noise_dataset_path=noise_train_dataset_path,
        ref_dataset_path=label_train_dataset_path,
        voc=voc,
        transform=Model.sentence_to_semi_char_tensor,
        label_transform=Model.sentence_to_word_tensor,
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, range(DATA_SIZE)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=Model.collate_fn,
        generator=torch.Generator(device=DEVICE),
        num_workers=LOADER_WORKER,
    )

    sclstm = Model.sclstm(
        word_size=WORD_SIZE,
        semi_char_vec_size=SEMI_CHAR_VEC_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYER,
        dropout=DROPOUT,
    )

    cross_entropy_loss = torch.nn.CrossEntropyLoss(
        ignore_index=voc["pad_token_idx"], label_smoothing=LABEL_SMOOTH
    )

    optimizer = torch.optim.Adam(sclstm.parameters())

    sclstm = torch.compile(sclstm)

    if not DEBUG:
        my_model, _ = Model.train(
            model=sclstm,
            loss_func=cross_entropy_loss,
            optimizer=optimizer,
            data_loader=train_loader,
            voc_fn=voc,
            num_epochs=NUM_EPOCHS,
            num_patience=PATIENCE,
        )

        MODEL_PATH = PATH.joinpath("sclstm.pth").absolute()
        torch.save(my_model, MODEL_PATH)
    else:
        Model.try_one_batch(
            model=sclstm,
            loss_func=cross_entropy_loss,
            optimizer=optimizer,
            data_loader=train_loader,
            voc_fn=voc,
            one_epoch=DEBUG_EPOCH,
        )
