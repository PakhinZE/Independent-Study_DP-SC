import math
import timeit
from collections import OrderedDict

from typing import List
from pathlib import Path
import torch
import torch.utils.data
import opacus.layers
import opacus.utils.batch_memory_manager


def readlines(readfile_path):
    sentences = open(readfile_path, encoding="utf-8").read().strip().split("\n")
    return [sentence for sentence in sentences]


def char_to_semi_char_vec(
    voc, char_size, semi_char_tensor, word_idx, char_list, char_count
):
    for char_idx, char in enumerate(char_list):
        # First
        if char_idx == 0:
            if char in voc["chartoken2idx"]:
                tensor_idx = (word_idx, voc["chartoken2idx"][char])
                semi_char_tensor[tensor_idx] = semi_char_tensor[tensor_idx] + 1
            else:
                tensor_idx = (word_idx, voc["char_unk_token_idx"])
                semi_char_tensor[tensor_idx] = semi_char_tensor[tensor_idx] + 1
        # Last
        elif char_idx == char_count - 1:
            pad_size = char_size * 2
            if char in voc["chartoken2idx"]:
                tensor_idx = (word_idx, voc["chartoken2idx"][char] + pad_size)
                semi_char_tensor[tensor_idx] = semi_char_tensor[tensor_idx] + 1
            else:
                tensor_idx = (word_idx, voc["char_unk_token_idx"] + pad_size)
                semi_char_tensor[tensor_idx] = semi_char_tensor[tensor_idx] + 1
        # Middle
        else:
            pad_size = char_size
            if char in voc["chartoken2idx"]:
                tensor_idx = (word_idx, voc["chartoken2idx"][char] + pad_size)
                semi_char_tensor[tensor_idx] = semi_char_tensor[tensor_idx] + 1
            else:
                tensor_idx = (word_idx, voc["char_unk_token_idx"] + pad_size)
                semi_char_tensor[tensor_idx] = semi_char_tensor[tensor_idx] + 1


def char_to_mask_semi_char_vec(
    voc, char_size, semi_char_tensor, word_idx, char_list, char_count
):
    for char_idx, char in enumerate(char_list):
        # First
        if char_idx == 0:
            tensor_idx = (word_idx, voc["char_mask_token_idx"])
            semi_char_tensor[tensor_idx] = 1
        # Last
        elif char_idx == char_count - 1:
            pad_size = char_size * 2
            tensor_idx = (word_idx, voc["char_mask_token_idx"] + pad_size)
            semi_char_tensor[tensor_idx] = 1
        # Middle
        else:
            pad_size = char_size
            tensor_idx = (word_idx, voc["char_mask_token_idx"] + pad_size)
            semi_char_tensor[tensor_idx] = 1


def sentence_to_semi_char_tensor(sentence: str, voc: dict, mask: float = 0):
    word_list = sentence.split()
    word_count = len(word_list)
    char_size = len(voc["chartoken2idx"])
    semi_char_vec_size = char_size * 3
    semi_char_tensor = torch.zeros(size=(word_count, semi_char_vec_size))
    word_mak_count = math.ceil(mask * word_count)
    word_nomak_count = word_count - word_mak_count
    for word_idx, word in enumerate(word_list):
        char_list = [*word]
        char_count = len(char_list)
        if word_idx < word_nomak_count:
            char_to_semi_char_vec(
                voc, char_size, semi_char_tensor, word_idx, char_list, char_count
            )
        else:
            char_to_mask_semi_char_vec(
                voc, char_size, semi_char_tensor, word_idx, char_list, char_count
            )
    return semi_char_tensor


def sentence_to_word_tensor(sentence: str, voc: dict):
    word_list = sentence.split()
    word_count = len(word_list)
    word_size = len(voc["token2idx"])
    word_tensor = torch.zeros(size=(word_count, word_size))
    for word_idx, word in enumerate(word_list):
        if word in voc["token2idx"]:
            tensor_idx = (word_idx, voc["token2idx"][word])
            word_tensor[tensor_idx] = 1
        else:
            tensor_idx = (word_idx, voc["unk_token_idx"])
            word_tensor[tensor_idx] = 1
    return word_tensor


def word_tensor_to_sentence(word_tensor: torch.Tensor, voc: dict):
    sentence = []
    index_list = torch.argmax(word_tensor, dim=1).tolist()
    if index_list:
        for idx in index_list:
            sentence.append(voc["idx2token"][idx])
    else:
        pass
    return " ".join(sentence)


class spell_correction_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        noise_dataset_path,
        ref_dataset_path,
        voc,
        transform=None,
        label_transform=None,
    ):
        self.noise_dataset = readlines(noise_dataset_path)
        self.label_dataset = readlines(ref_dataset_path)
        self.voc = voc
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.noise_dataset)

    def __getitem__(self, idx):
        data = self.noise_dataset[idx]
        label = self.label_dataset[idx]
        if self.transform:
            data = self.transform(data, self.voc)
        if self.label_transform:
            label = self.label_transform(label, self.voc)
        return data, label


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    batch_data, batch_label = zip(*batch)

    length_batch_data = [tensor.size(0) for tensor in batch_data]
    batch_data = torch.nn.utils.rnn.pack_padded_sequence(
        torch.nn.utils.rnn.pad_sequence(list(batch_data), batch_first=True),
        lengths=length_batch_data,  # type: ignore
        enforce_sorted=False,
        batch_first=True,
    )

    length_batch_label = [tensor.size(0) for tensor in batch_label]
    batch_label = [torch.argmax(label, dim=1) for label in batch_label]
    batch_label = torch.nn.utils.rnn.pack_padded_sequence(
        torch.nn.utils.rnn.pad_sequence(batch_label, batch_first=True),
        lengths=length_batch_label,  # type: ignore
        enforce_sorted=False,
        batch_first=True,
    )

    return batch_data, batch_label


class sclstm(torch.nn.Module):
    def __init__(
        self,
        word_size: int,
        semi_char_vec_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.word_size = word_size
        self.semi_char_vec_size = semi_char_vec_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = torch.nn.LSTM(
            input_size=self.semi_char_vec_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(self.hidden_size * 2, self.word_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.nn.utils.rnn.unpack_sequence(x)  # type: ignore
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        x = self.linear(x)
        return x


class dp_sclstm(torch.nn.Module):
    def __init__(
        self,
        word_size: int,
        semi_char_vec_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.word_size = word_size
        self.semi_char_vec_siz = semi_char_vec_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = opacus.layers.dp_rnn.DPLSTM(
            input_size=self.semi_char_vec_siz,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(self.hidden_size * 2, self.word_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.nn.utils.rnn.unpack_sequence(x)  # type: ignore
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        x = self.linear(x)
        return x


def train(
    model,
    loss_func,
    optimizer,
    data_loader,
    voc_fn: dict | None = None,
    num_epochs: int = 1,
    num_patience: int = 0,
    checkpoint: List[Path] = [Path(), Path(), Path()],
):
    begin_t = timeit.default_timer()
    have_patience = True if num_patience > 0 else False
    patience = 0
    best_model = None
    best_loss = float("Inf")
    pad_value = voc_fn["pad_token_idx"] if voc_fn else 0.0
    num_batch = len(data_loader)

    print(f"Number of Epochs: {num_epochs}")
    print(f"Number of Batch {num_batch}")

    for epoch in range(num_epochs):
        begin_t_epoch = timeit.default_timer()

        epoch_loss = 0

        model.train()
        for x_batch, y_batch in data_loader:
            y_logit_pred = model(x_batch)
            y_logit_pred = torch.movedim(y_logit_pred, 2, 1)

            y_batch = torch.nn.utils.rnn.unpack_sequence(y_batch)  # type: ignore
            y_batch = torch.nn.utils.rnn.pad_sequence(
                y_batch, batch_first=True, padding_value=pad_value
            )

            loss = loss_func(y_logit_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.float()

        epoch_loss = epoch_loss / num_batch
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            if have_patience:
                patience = 0
        else:
            if have_patience:
                patience = patience + 1

        end_t_epoch = timeit.default_timer()

        if all(
            [
                not checkpoint[0] == Path(),
                not checkpoint[1] == Path(),
                not checkpoint[2] == Path(),
            ]
        ):
            checkpoint_path = checkpoint[0]
            checkpoint_temp_path = checkpoint[1]
            checkpoint_pass_path = checkpoint[2]
            save_checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            }
            torch.save(save_checkpoint, checkpoint_temp_path)
            checkpoint_pass_path.touch()
            checkpoint_temp_path.replace(checkpoint_path)
            checkpoint_pass_path.unlink()

        print(f"Current Epoch: {epoch + 1}")
        print(f"Epoch Time Spent: {end_t_epoch - begin_t_epoch}")
        print(f"Cross-entropy: {epoch_loss}")

        if have_patience:
            if patience == num_patience:
                break

    end_t = timeit.default_timer()
    print(f"Best Cross-entropy: {best_loss}")
    print(f"Total Time Spent: {end_t - begin_t}")
    print("!!!Complete all epoch!!!")
    return best_model, best_loss


def try_one_batch(
    model,
    loss_func,
    optimizer,
    data_loader,
    voc_fn: dict | None = None,
    one_epoch: bool = False,
):
    pad_value = voc_fn["pad_token_idx"] if voc_fn else 0.0

    begin_t = timeit.default_timer()

    model.train()
    for x_batch, y_batch in data_loader:
        y_logit_pred = model(x_batch)
        y_logit_pred = torch.movedim(y_logit_pred, 2, 1)

        y_batch = torch.nn.utils.rnn.unpack_sequence(y_batch)  # type: ignore
        y_batch = torch.nn.utils.rnn.pad_sequence(
            y_batch, batch_first=True, padding_value=pad_value
        )

        loss = loss_func(y_logit_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not one_epoch:
            break

    end_t = timeit.default_timer()
    if one_epoch:
        print(f"Epoch Time Spent: {end_t - begin_t}")
    else:
        print(f"Batch Time Spent: {end_t - begin_t}")


def dp_train(
    model,
    loss_func,
    optimizer,
    data_loader,
    privacy_engine,
    voc_fn: dict | None = None,
    num_epochs: int = 1,
    num_patience: int = 0,
    max_physical_batch_size: int = 1,
    checkpoint: List[Path] = [Path(), Path(), Path()],
):
    begin_t = timeit.default_timer()
    have_patience = True if num_patience > 0 else False
    patience = 0
    best_model = None
    best_loss = float("Inf")
    pad_value = voc_fn["pad_token_idx"] if voc_fn else 0.0
    expt_num_batch = len(data_loader)

    print(f"Number of Epochs: {num_epochs}")
    print(f"Expected Number of Batch {expt_num_batch}")

    for epoch in range(num_epochs):
        begin_t_epoch = timeit.default_timer()

        epoch_loss = 0

        model.train()
        with opacus.utils.batch_memory_manager.BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_data_loader:
            expt_num_microbatch = len(memory_data_loader)
            print(f"Expected Number of Micro Batch {expt_num_microbatch}")
            for x_batch, y_batch in memory_data_loader:
                y_logit_pred = model(x_batch)
                y_logit_pred = torch.movedim(y_logit_pred, 2, 1)

                y_batch = torch.nn.utils.rnn.unpack_sequence(y_batch)  # type: ignore
                y_batch = torch.nn.utils.rnn.pad_sequence(
                    y_batch, batch_first=True, padding_value=pad_value
                )

                loss = loss_func(y_logit_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss = epoch_loss + loss.float()

        epoch_loss = epoch_loss / expt_num_batch
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            if have_patience:
                patience = 0
        else:
            if have_patience:
                patience = patience + 1

        end_t_epoch = timeit.default_timer()

        if all(
            [
                not checkpoint[0] == Path(),
                not checkpoint[1] == Path(),
                not checkpoint[2] == Path(),
            ]
        ):
            checkpoint_path = checkpoint[0]
            checkpoint_temp_path = checkpoint[1]
            checkpoint_pass_path = checkpoint[2]
            accountant = dict()
            privacy_engine.accountant.state_dict(accountant)
            save_checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "accountant": accountant,
                "epoch": epoch + 1,
            }
            torch.save(save_checkpoint, checkpoint_temp_path)
            checkpoint_pass_path.touch()
            checkpoint_temp_path.replace(checkpoint_path)
            checkpoint_pass_path.unlink()

        print(f"Current Epoch: {epoch + 1}")
        print(f"Epoch Time Spent: {end_t_epoch - begin_t_epoch}")
        print(f"Cross-entropy: {epoch_loss}")

        if have_patience:
            if patience == num_patience:
                break

    end_t = timeit.default_timer()
    print(f"Best Cross-entropy: {best_loss}")
    print(f"Total Time Spent: {end_t - begin_t}")
    print("!!!Complete all epoch!!!")
    return best_model, best_loss


def dp_try_one_batch(
    model,
    loss_func,
    optimizer,
    data_loader,
    voc_fn: dict | None = None,
    max_physical_batch_size: int = 1,
    one_epoch: bool = False,
):
    pad_value = voc_fn["pad_token_idx"] if voc_fn else 0.0

    begin_t = timeit.default_timer()

    model.train()
    with opacus.utils.batch_memory_manager.BatchMemoryManager(
        data_loader=data_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer,
    ) as memory_data_loader:
        print(f"Number of Micro Batch {len(memory_data_loader)}")
        for x_batch, y_batch in memory_data_loader:
            y_logit_pred = model(x_batch)
            y_logit_pred = torch.movedim(y_logit_pred, 2, 1)

            y_batch = torch.nn.utils.rnn.unpack_sequence(y_batch)  # type: ignore
            y_batch = torch.nn.utils.rnn.pad_sequence(
                y_batch, batch_first=True, padding_value=pad_value
            )

            loss = loss_func(y_logit_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not one_epoch:
                break

    end_t = timeit.default_timer()
    if one_epoch:
        print(f"Epoch Time Spent: {end_t - begin_t}")
    else:
        print(f"Batch Time Spent: {end_t - begin_t}")


def load_compiled_model(compiled_model_path):  # No longer used
    compiled_model = torch.load(compiled_model_path)
    model_weight = OrderedDict()
    for key, value in compiled_model.items():
        if key.split("_orig_mod.")[0]:
            model_weight.update({key: value})
        else:
            new_key = key.split("_orig_mod.")[1]
            model_weight.update({new_key: value})
    return model_weight


def load_dp_model(dp_model_path):  # No longer used
    dp_model = torch.load(dp_model_path)
    model_weight = OrderedDict()
    for key, value in dp_model.items():
        if key.split("_orig_mod._module.")[0]:
            model_weight.update({key: value})
        else:
            new_key = key.split("_orig_mod._module.")[1]
            model_weight.update({new_key: value})
    return model_weight


if __name__ == "__main__":
    pass
