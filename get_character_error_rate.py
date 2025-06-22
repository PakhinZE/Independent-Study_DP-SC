# %%
from pathlib import Path
import torch
from torchmetrics.text import CharErrorRate

import model

# %%
PATH = Path(__file__).parent
DEVICE = "cuda"

# %%
torch.set_default_device(DEVICE)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# %%
label_path = (
    PATH.joinpath("normalization_dataset/")
    .joinpath("normalization_test.1blm")
    .absolute()
)
mask_label_path = (
    PATH.joinpath("normalization_dataset/")
    .joinpath("normalization_train.1blm")
    .absolute()
)

# %%
pred_path = PATH.joinpath("result").joinpath("prediction").absolute()
dp_pred_path = PATH.joinpath("result").joinpath("dp-prediction").absolute()
mask_pred_path = PATH.joinpath("result").joinpath("mask-prediction").absolute()
dp_mask_pred_path = PATH.joinpath("result").joinpath("dp-mask-prediction").absolute()

# %%
char_error_rate = CharErrorRate()

# %%
labels = model.readlines(label_path)

# %%
preds = model.readlines(pred_path)
score = char_error_rate(preds, labels)
print(f"Score For Spell Correction: {score}")
del preds

# %%
dp_preds = model.readlines(dp_pred_path)
dp_score = char_error_rate(dp_preds, labels)
print(f"Score For DP Spell Correction: {dp_score}")
del dp_preds

# %%
del labels

# %%
mask_labels = model.readlines(mask_label_path)

# %%
mask_preds = model.readlines(mask_pred_path)
mask_score = char_error_rate(mask_preds, mask_labels)
print(f"Score For Memoization: {mask_score}")
del mask_preds
# %%
dp_mask_preds = model.readlines(dp_mask_pred_path)
dp_mask_score = char_error_rate(dp_mask_preds, mask_labels)
print(f"Score For DP Memoization: {dp_mask_score}")
del dp_mask_preds

# %%
del mask_labels
