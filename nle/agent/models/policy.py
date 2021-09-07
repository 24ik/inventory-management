import mmle.nn as mnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePolicy(nn.Module):
    def __init__(self, num_action, hidden_dim):
        super().__init__()

        self._fc = mnn.FC(
            hidden_dim, num_action, bn=False, activ="log_softmax", activ_kwargs={"dim": -1}
        )

    def forward(self, feature, item_score):
        """
        (B, H), _ -> (B, A)
        """

        log_prob = self._fc(feature)  # (B, A)

        return log_prob, log_prob, None  # (B, A), (B, A), _


class MetaPolicy(nn.Module):
    def __init__(self, num_virtual_action, hidden_dim):
        super().__init__()

        self._fc = mnn.FC(
            hidden_dim, num_virtual_action, bn=False, activ="log_softmax", activ_kwargs={"dim": -1}
        )

    def forward(self, feature, item_score):
        """
        (B, H), (B, N) -> (B, A)
        """

        virtual_log_prob = self._fc(feature)  # (B, a+1)
        item_log_prob = F.log_softmax(item_score, dim=1)  # (B, N)
        policy_log_prob = torch.cat(
            [
                virtual_log_prob[:, :-1],
                virtual_log_prob[:, -1:] + item_log_prob,
            ],
            dim=-1,
        )  # (B, A)

        return policy_log_prob, virtual_log_prob, item_log_prob  # (B, A), (B, a+1), (B, N)


def make_policy_model(flags, num_action, num_virtual_action):
    model_kind = flags.model.policy.model
    if model_kind == "baseline":
        return BasePolicy(num_action, flags.model.hidden_dim)
    elif model_kind == "meta":
        return MetaPolicy(num_virtual_action, flags.model.hidden_dim)
    else:
        raise ValueError(f"policy.model == {model_kind}")
