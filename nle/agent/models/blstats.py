import mmle
import mmle.nn as mnn
import torch
import torch.nn as nn

from nle.nethack import BLSTATS_SHAPE


class BlstatsBaseModel(nn.Module):
    def __init__(self, use_idxes, hidden_dim):
        super().__init__()

        if use_idxes is None:
            self._use_idxes = None
            self._preprocess = mmle.id_fn
            input_dim = BLSTATS_SHAPE[0]
        else:
            self.register_buffer("_use_idxes", use_idxes.detach().clone())
            self._preprocess = lambda x: x.index_select(1, self._use_idxes)
            (input_dim,) = use_idxes.shape

        self._mlp = nn.Sequential(
            mnn.FC(input_dim, hidden_dim, bn=False),
            mnn.FC(hidden_dim, hidden_dim, bn=False),
        )

    def forward(self, x):
        """
        (B, S) -> (B, E)
        """

        return self._mlp(self._preprocess(x))  # (B, E)


def make_blstats_model(flags):
    blstats = flags.model.blstats

    if blstats.use_info == "useful":
        use_idx = torch.tensor(
            [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22]
        )
    else:
        use_idx = None

    model_kind = blstats.model
    if model_kind == "none":
        return None, 0
    elif model_kind == "baseline":
        return BlstatsBaseModel(use_idx, blstats.hidden_dim), blstats.hidden_dim
    else:
        raise ValueError(f"blstats.model == {model_kind}")
