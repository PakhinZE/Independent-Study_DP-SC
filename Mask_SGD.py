import pickle
from pathlib import Path
from functools import partial

import torch
import torch.utils.data

import model


if __name__ == "__main__":
    PATH = Path(__file__).parent
    DEVICE = "cuda"

    # Data Size
    DATA_SIZE = None  # None
    # Mask
    MASK = 0.4
    # Loader
    LOADER_WORKER = 0  # Default = 0
    BATCH_SIZE = 32
    # NN
    HIDDEN_SIZE = 512
    NUM_LAYER = 2
    DROPOUT = 0.4
    # Train
    NUM_EPOCHS = 100
    PATIENCE = 3
    LABEL_SMOOTH = 0.1

    # Checkpoint
    CHECKPOINT = True
    if CHECKPOINT:
        checkpoint_path = (
            PATH.joinpath("checkpoint")
            .joinpath(Path(__file__).stem)
            .with_suffix(".pth")
            .absolute()
        )
        checkpoint_temp_path = (
            PATH.joinpath("checkpoint")
            .joinpath(Path(__file__).stem)
            .with_suffix(".temp")
            .absolute()
        )
        checkpoint_pass_path = checkpoint_path.with_suffix(".pass")
    else:
        checkpoint_path = Path()
        checkpoint_temp_path = Path()
        checkpoint_pass_path = Path()

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

    sentence_to_semi_char_tensor = partial(
        model.sentence_to_semi_char_tensor, mask=MASK
    )

    train_dataset = model.spell_correction_dataset(
        noise_dataset_path=label_train_dataset_path,  # no noise
        ref_dataset_path=label_train_dataset_path,
        voc=voc,
        transform=sentence_to_semi_char_tensor,  # mask
        label_transform=model.sentence_to_word_tensor,
    )

    if DATA_SIZE:
        loading_datset = torch.utils.data.Subset(train_dataset, range(DATA_SIZE))
    else:
        loading_datset = train_dataset
    train_loader = torch.utils.data.DataLoader(
        loading_datset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=model.collate_fn,
        generator=torch.Generator(device=DEVICE),
        num_workers=LOADER_WORKER,
    )

    cross_entropy_loss = torch.nn.CrossEntropyLoss(
        ignore_index=voc["pad_token_idx"], label_smoothing=LABEL_SMOOTH
    )

    sclstm = model.sclstm(
        word_size=WORD_SIZE,
        semi_char_vec_size=SEMI_CHAR_VEC_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYER,
        dropout=DROPOUT,
    )

    optimizer = torch.optim.Adam(sclstm.parameters())

    sclstm = torch.compile(sclstm)

    if CHECKPOINT and (
        checkpoint_temp_path.is_file() and checkpoint_pass_path.is_file()
    ):
        checkpoint_temp_path.replace(checkpoint_path)

    if CHECKPOINT and checkpoint_path.is_file():
        save_checkpoint = torch.load(checkpoint_path, weights_only=False)

        NUM_EPOCHS = NUM_EPOCHS - save_checkpoint["epoch"]
        if NUM_EPOCHS == 0:
            print("No remain epoch in checkpoint")
            exit()
        sclstm.load_state_dict(save_checkpoint["model"])
        optimizer.load_state_dict(save_checkpoint["optimizer"])

    if not DEBUG:
        my_model, _ = model.train(
            model=sclstm,
            loss_func=cross_entropy_loss,
            optimizer=optimizer,
            data_loader=train_loader,
            voc_fn=voc,
            num_epochs=NUM_EPOCHS,
            num_patience=PATIENCE,
            checkpoint=[checkpoint_path, checkpoint_temp_path, checkpoint_pass_path],
        )

        model_path = PATH.joinpath("model").joinpath("mask-sclstm.pth").absolute()
        torch.save(my_model, model_path)
    else:
        model.try_one_batch(
            model=sclstm,
            loss_func=cross_entropy_loss,
            optimizer=optimizer,
            data_loader=train_loader,
            voc_fn=voc,
            one_epoch=DEBUG_EPOCH,
        )
