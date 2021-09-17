"""
Official code of "Keisuke Izumiya and Edgar Simo-Serra, Inventory Management with Attention-Based Meta Actions, IEEE Conference on Games (CoG), 2021."
    Copyright (C) 2021 Keisuke Izumiya

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
