# %%
import os
import pickle
from pathlib import Path
from tqdm import tqdm

# %%
PATH = Path(__file__).parent
print(PATH)
data_dir = PATH.joinpath("normalization_dataset")
train_clean_path = data_dir.joinpath("normalization_train.1blm").absolute()
train_corrupt_path = data_dir.joinpath(
    "normalization_train.1blm.noise.random"
).absolute()
test_clean_path = data_dir.joinpath("normalization_test.1blm").absolute()
test_corrupt_path = data_dir.joinpath("normalization_test.1blm.noise.random").absolute()


# %%
def load_data(base_path, corr_file, incorr_file):
    # load files
    if base_path:
        assert os.path.exists(base_path)
    incorr_data = []
    opfile1 = open(os.path.join(base_path, incorr_file), "r")
    for line in opfile1:
        if line.strip() != "":
            incorr_data.append(line.strip())
    opfile1.close()
    corr_data = []
    opfile2 = open(os.path.join(base_path, corr_file), "r")
    for line in opfile2:
        if line.strip() != "":
            corr_data.append(line.strip())
    opfile2.close()
    assert len(incorr_data) == len(corr_data)

    # verify if token split is same
    for i, (x, y) in tqdm(enumerate(zip(corr_data, incorr_data))):
        x_split, y_split = x.split(), y.split()
        try:
            assert len(x_split) == len(y_split)
        except AssertionError:
            print(
                "# tokens in corr and incorr mismatch. retaining and trimming to min len."
            )
            print(x_split, y_split)
            mn = min([len(x_split), len(y_split)])
            corr_data[i] = " ".join(x_split[:mn])
            incorr_data[i] = " ".join(y_split[:mn])
            print(corr_data[i], incorr_data[i])

    # return as pairs
    data = []
    for x, y in tqdm(zip(corr_data, incorr_data)):
        data.append((x, y))

    print(f"loaded tuples of (corr,incorr) examples from {base_path}")
    return data


# %%
def get_char_tokens(use_default: bool, data=None):
    if not use_default and data is None:
        raise Exception("data is None")

    # reset char token utils
    chartoken2idx, idx2chartoken = {}, {}
    # (
    #     char_unk_token,
    #     char_pad_token,
    #     char_start_token,
    #     char_end_token,
    #     char_mask_token,
    # ) = "<<CHAR_UNK>>", "<<CHAR_PAD>>", "<<CHAR_START>>", "<<CHAR_END>>", "<<CHAR_MAK>>"
    # special_tokens = [
    #     char_unk_token,
    #     char_pad_token,
    #     char_start_token,
    #     char_end_token,
    #     char_mask_token,
    # ]
    # for char in special_tokens:
    #     idx = len(chartoken2idx)
    #     chartoken2idx[char] = idx
    #     idx2chartoken[idx] = char

    if use_default:
        chars = list(
            """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""
        )
        for char in chars:
            if char not in chartoken2idx:
                idx = len(chartoken2idx)
                chartoken2idx[char] = idx
                idx2chartoken[idx] = char
    else:
        # helper funcs
        # isascii = lambda s: len(s) == len(s.encode())
        """
        # load batches of lines and obtain unique chars
        nlines = len(data)
        bsize = 5000
        nbatches = int( np.ceil(nlines/bsize) )
        for i in tqdm(range(nbatches)):
            blines = " ".join( [ex for ex in data[i*bsize:(i+1)*bsize]] )
            #bchars = set(list(blines))
            for char in bchars:
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char
        """
        # realized that set doesn't preserve order!!
        for line in tqdm(data):
            for char in line:  # type: ignore
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char
        (
            char_unk_token,
            char_pad_token,
            char_start_token,
            char_end_token,
            char_mask_token,
        ) = (
            "<<CHAR_UNK>>",
            "<<CHAR_PAD>>",
            "<<CHAR_START>>",
            "<<CHAR_END>>",
            "<<CHAR_MAK>>",
        )
        special_tokens = [
            char_unk_token,
            char_pad_token,
            char_start_token,
            char_end_token,
            char_mask_token,
        ]
        for char in special_tokens:
            idx = len(chartoken2idx)
            chartoken2idx[char] = idx
            idx2chartoken[idx] = char

    print(f"number of unique chars found: {len(chartoken2idx)}")
    print(chartoken2idx)
    return_dict = {}
    return_dict["chartoken2idx"] = chartoken2idx
    return_dict["idx2chartoken"] = idx2chartoken
    return_dict["char_unk_token"] = char_unk_token
    return_dict["char_pad_token"] = char_pad_token
    return_dict["char_start_token"] = char_start_token
    return_dict["char_end_token"] = char_end_token
    return_dict["char_maks_token"] = char_mask_token
    # new
    return_dict["char_unk_token_idx"] = chartoken2idx[char_unk_token]
    return_dict["char_pad_token_idx"] = chartoken2idx[char_pad_token]
    return_dict["char_start_token_idx"] = chartoken2idx[char_start_token]
    return_dict["char_end_token_idx"] = chartoken2idx[char_end_token]
    return_dict["char_mask_token_idx"] = chartoken2idx[char_mask_token]

    return return_dict


# %%
def get_tokens(
    data,
    keep_simple=False,
    min_max_freq=(1, float("inf")),
    topk=None,
    intersect=[],
    load_char_tokens=False,
):
    # get all tokens
    token_freq, token2idx, idx2token = {}, {}, {}
    for example in tqdm(data):  ##########################
        for token in example.split():
            if token not in token_freq:
                token_freq[token] = 0
            token_freq[token] += 1
    print(f"Total tokens found: {len(token_freq)}")

    # retain only simple tokens
    if keep_simple:

        def isascii(s):
            return len(s) == len(s.encode())

        def hasdigits(s):
            return len([x for x in list(s) if x.isdigit()]) > 0

        tf = [
            (t, f)
            for t, f in [*token_freq.items()]
            if (isascii(t) and not hasdigits(t))
        ]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only tokens with specified min and max range
    if min_max_freq[0] > 1 or min_max_freq[1] < float("inf"):
        sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
        tf = [
            (i[0], i[1])
            for i in sorted_
            if (i[1] >= min_max_freq[0] and i[1] <= min_max_freq[1])
        ]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only topk tokens
    if topk is not None:
        sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
        token_freq = {t: f for (t, f) in list(sorted_)[:topk]}
        print(f"Total tokens retained: {len(token_freq)}")

    # retain only interection of tokens
    if len(intersect) > 0:
        tf = [
            (t, f)
            for t, f in [*token_freq.items()]
            if (t in intersect or t.lower() in intersect)
        ]
        token_freq = {t: f for (t, f) in tf}
        print(f"Total tokens retained: {len(token_freq)}")

    # create token2idx and idx2token
    for token in token_freq:
        idx = len(token2idx)
        idx2token[idx] = token
        token2idx[token] = idx

    # add <<PAD>> special token
    ntokens = len(token2idx)
    pad_token = "<<PAD>>"
    token_freq.update({pad_token: -1})
    token2idx.update({pad_token: ntokens})
    idx2token.update({ntokens: pad_token})

    # add <<UNK>> special token
    ntokens = len(token2idx)
    unk_token = "<<UNK>>"
    token_freq.update({unk_token: -1})
    token2idx.update({unk_token: ntokens})
    idx2token.update({ntokens: unk_token})

    # new
    # add <<EOS>> special token
    ntokens = len(token2idx)
    eos_token = "<<EOS>>"
    token_freq.update({eos_token: -1})
    token2idx.update({eos_token: ntokens})
    idx2token.update({ntokens: eos_token})

    # new
    # add <<MAK>> special token
    ntokens = len(token2idx)
    mak_token = "<<MAK>>"
    token_freq.update({mak_token: -1})
    token2idx.update({mak_token: ntokens})
    idx2token.update({ntokens: mak_token})

    # return dict
    token_freq = list(
        sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
    )
    return_dict = {
        "token2idx": token2idx,
        "idx2token": idx2token,
        "token_freq": token_freq,
        "pad_token": pad_token,
        "unk_token": unk_token,
        "eos_token": eos_token,
        "mak_token": mak_token,
    }
    # new
    return_dict.update(
        {
            "pad_token_idx": token2idx[pad_token],
            "unk_token_idx": token2idx[unk_token],
            "eos_token_idx": token2idx[eos_token],
            "mak_token_idx": token2idx[mak_token],
        }
    )

    # load_char_tokens
    if load_char_tokens:
        print("loading character tokens")
        char_return_dict = get_char_tokens(use_default=False, data=data)
        return_dict.update(char_return_dict)

    return return_dict


# %%
def save_vocab_dict(path_: str, vocab_: dict):
    """
    path_: path where the vocab pickle file to be saved
    vocab_: the dict data
    """
    with open(path_, "wb") as fp:
        pickle.dump(vocab_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return


# %%
print("------------------------")
train_data = load_data(data_dir, train_clean_path, train_corrupt_path)
# test_data = load_data(data_dir, test_clean_path, test_corrupt_path)
print("------------------------")
clean_sentences = [sentence_pair[0] for sentence_pair in train_data]
corrupt_sentences = [sentence_pair[1] for sentence_pair in train_data]
voc = get_tokens(
    clean_sentences,
    keep_simple=False,
    min_max_freq=(1, float("inf")),
    topk=100000,
    load_char_tokens=True,
)
print("------------------------")

# %%
voc.keys()

# %%
len(voc["token2idx"])

# %%
len(voc["chartoken2idx"])

# %%
voc_path = PATH.joinpath("voc").absolute()

# %%
save_vocab_dict(str(voc_path), voc)
