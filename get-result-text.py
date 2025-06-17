# %%
import pickle
from pathlib import Path
from tqdm import tqdm
from functools import partial


import torch
import torch.utils.data
import opacus

import model

# %%
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
NUM_LAYER = 2  #
DROPOUT = 0.4
# Train
NUM_EPOCHS = 100
PATIENCE = 3
LABEL_SMOOTH = 0.1
# Debug
DEBUG = False
DEBUG_ONE = False

# %%
torch.set_default_device(DEVICE)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# %%
voc_path = PATH.joinpath("voc").absolute()
with open(voc_path, "rb") as voc_file:
    voc = pickle.load(voc_file)
WORD_SIZE = len(voc["token2idx"])
CHAR_SIZE = len(voc["chartoken2idx"])
SEMI_CHAR_VEC_SIZE = CHAR_SIZE * 3

# %%
noise_train_dataset_path = PATH.joinpath("normalization_dataset/").joinpath(
    "normalization_train.1blm.noise.random"
)
label_train_dataset_path = PATH.joinpath("normalization_dataset/").joinpath(
    "normalization_train.1blm"
)
noise_test_dataset_path = PATH.joinpath("normalization_dataset/").joinpath(
    "normalization_test.1blm.noise.random"
)
label_test_dataset_path = PATH.joinpath("normalization_dataset/").joinpath(
    "normalization_test.1blm"
)

# %%
test_dataset = model.spell_correction_dataset(
    noise_dataset_path=noise_test_dataset_path,
    ref_dataset_path=label_test_dataset_path,
    voc=voc,
    transform=model.sentence_to_semi_char_tensor,
    label_transform=model.sentence_to_word_tensor,
)
if DATA_SIZE:
    loading_datset = torch.utils.data.Subset(test_dataset, range(DATA_SIZE))
else:
    loading_datset = test_dataset
test_loader = torch.utils.data.DataLoader(
    loading_datset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=model.collate_fn,
    generator=torch.Generator(device=DEVICE),
    num_workers=LOADER_WORKER,
)

sentence_to_semi_char_tensor = partial(model.sentence_to_semi_char_tensor, mask=MASK)
mask_test_dataset = model.spell_correction_dataset(
    noise_dataset_path=label_train_dataset_path,  # no noise
    ref_dataset_path=label_train_dataset_path,
    voc=voc,
    transform=sentence_to_semi_char_tensor,  # mask
    label_transform=model.sentence_to_word_tensor,
)
if DATA_SIZE:
    loading_datset = torch.utils.data.Subset(mask_test_dataset, range(DATA_SIZE))
else:
    loading_datset = mask_test_dataset
mask_test_loader = torch.utils.data.DataLoader(
    loading_datset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=model.collate_fn,
    generator=torch.Generator(device=DEVICE),
    num_workers=LOADER_WORKER,
)

# %%
sclstm = model.sclstm(
    word_size=WORD_SIZE,
    semi_char_vec_size=SEMI_CHAR_VEC_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)
mask_sclstm = model.sclstm(
    word_size=WORD_SIZE,
    semi_char_vec_size=SEMI_CHAR_VEC_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)
dp_sclstm = model.dp_sclstm(
    word_size=WORD_SIZE,
    semi_char_vec_size=SEMI_CHAR_VEC_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)
dp_mask_sclstm = model.dp_sclstm(
    word_size=WORD_SIZE,
    semi_char_vec_size=SEMI_CHAR_VEC_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)

# %%
model_path = PATH.joinpath("model").joinpath("sclstm.pth").absolute()
dp_model_path = PATH.joinpath("model").joinpath("dp-sclstm.pth").absolute()
mask_model_path = PATH.joinpath("model").joinpath("mask-sclstm.pth").absolute()
dp_mask_model_path = PATH.joinpath("model").joinpath("dp-mask-sclstm.pth").absolute()

# %%
sclstm = torch.compile(sclstm)
sclstm.load_state_dict(torch.load(model_path))

# %%
dp_sclstm = opacus.grad_sample.grad_sample_module.GradSampleModule(dp_sclstm)
dp_sclstm.load_state_dict(torch.load(dp_model_path))

# %%
mask_sclstm = torch.compile(mask_sclstm)
mask_sclstm.load_state_dict(torch.load(mask_model_path))

# %%
dp_mask_sclstm = opacus.grad_sample.grad_sample_module.GradSampleModule(dp_mask_sclstm)
dp_mask_sclstm.load_state_dict(torch.load(dp_mask_model_path))


# %%
def get_predict_text(model, data_loader, voc_fn):
    model.eval()
    text_preds = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(data_loader):
            y_logit_pred = model(x_batch)
            y_logit_pred = y_logit_pred[0]
            text_pred = model.word_tensor_to_sentence(y_logit_pred, voc_fn)
            text_preds.append(text_pred)

            if DEBUG:
                print(text_pred)

            if DEBUG_ONE:
                break

    return text_preds


# %%
def write_to_file(list_string, path):
    file = open(path, "w")
    for string in list_string:
        file.write(string)
        file.write("\n")
    file.close()


# %%
preds = get_predict_text(model=sclstm, data_loader=test_loader, voc_fn=voc)
pred_path = PATH.joinpath("result").joinpath("prediction").absolute()
write_to_file(preds, pred_path)
del preds

# %%
dp_preds = get_predict_text(model=dp_sclstm, data_loader=test_loader, voc_fn=voc)
dp_pred_path = PATH.joinpath("result").joinpath("dp-prediction").absolute()
write_to_file(dp_preds, dp_pred_path)
del dp_preds

# %%
mask_preds = get_predict_text(
    model=mask_sclstm, data_loader=mask_test_loader, voc_fn=voc
)
mask_pred_path = PATH.joinpath("result").joinpath("mask-prediction").absolute()
write_to_file(mask_preds, mask_pred_path)
del mask_preds

# %%
dp_mask_preds = get_predict_text(
    model=dp_mask_sclstm, data_loader=mask_test_loader, voc_fn=voc
)
dp_mask_pred_path = PATH.joinpath("result").joinpath("dp-mask-prediction").absolute()
write_to_file(dp_mask_preds, dp_mask_pred_path)
del dp_mask_preds
