from nle.agent.models.txt import TxtCnn, TxtRnn
from nle.nethack import MESSAGE_SHAPE


def make_msg_model(flags):
    msg = flags.model.msg
    txt = flags.model.txt

    model_kind = msg.model
    if model_kind == "none":
        return None, 0
    elif model_kind == "cnn":
        return (
            TxtCnn(
                txt.emb_dim,
                txt.emb_kind,
                msg.hidden_dim,
                msg.cnn.depth,
                MESSAGE_SHAPE[0],
                flags.model.use_index_select,
            ),
            msg.hidden_dim,
        )
    elif model_kind in ("gru", "lstm"):
        return (
            TxtRnn(
                model_kind, txt.emb_dim, txt.emb_kind, msg.hidden_dim, flags.model.use_index_select
            ),
            msg.hidden_dim,
        )
    else:
        raise ValueError(f"msg.model == {model_kind}")
