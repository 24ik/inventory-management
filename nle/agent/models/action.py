import torch.nn as nn

from nle.agent.models.util import MyEmbedding


class ActionBaseModel(nn.Module):
    def __init__(self, num_action, emb_dim, use_index_select):
        super().__init__()

        self._emb = MyEmbedding(num_action, emb_dim, use_index_select)

    def forward(self, x):
        """
        (B, A) -> (B, H)
        """

        return self._emb(x)


def make_action_model(flags, num_action):
    model_kind = flags.model.action.model
    if model_kind == "none":
        return None, 0
    elif model_kind == "baseline":
        return (
            ActionBaseModel(num_action, flags.model.action.emb_dim, flags.model.use_index_select),
            flags.model.action.emb_dim,
        )
    else:
        raise ValueError(f"action.model == {model_kind}")
