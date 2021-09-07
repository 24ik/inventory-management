# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from typing import NamedTuple, Union

import torch
from torch import nn

from nle import nethack as nh
from nle.agent.models.util import MyEmbedding
from nle.agent.util.id_pairs import id_pairs_table

Ratio = Union[int, bool]


class Targets(NamedTuple):
    """Class for configuring whch ids you want to embed into the single
    GlyphEmbedding, and in what ratios. The ratio is only relevant if
    do_linear_layer is false, and the embedding is pure concatenation.
    """

    glyphs: Ratio = 0
    groups: Ratio = 0
    subgroup_ids: Ratio = 0
    colors: Ratio = 0
    chars: Ratio = 0
    specials: Ratio = 0
    do_linear_layer: bool = True

    def count_matrices(self):
        """Count of matrices required"""
        return sum(self) - int(self.do_linear_layer)


GLYPH_TYPE_STRATEGIES = {
    "full": Targets(glyphs=True),
    "group_id": Targets(groups=True, subgroup_ids=True),
    "color_char": Targets(colors=True, chars=True, specials=True),
    "all": Targets(groups=True, subgroup_ids=True, colors=True, chars=True, specials=True),
    "all_cat": Targets(
        groups=1, subgroup_ids=3, colors=1, chars=2, specials=1, do_linear_layer=False
    ),
}


class GlyphEmbedding(nn.Module):
    """Take the glyph information and return an embedding vector."""

    def __init__(self, glyph_type, dimension, use_index_select=None):
        super(GlyphEmbedding, self).__init__()
        self.glyph_type = glyph_type
        self.dim = dimension

        if glyph_type not in GLYPH_TYPE_STRATEGIES:
            raise RuntimeError("unexpected glyph_type=%s" % self.glyph_type)
        strategy = GLYPH_TYPE_STRATEGIES[glyph_type]
        self.strategy = strategy

        self._unit_dim = dimension // strategy.count_matrices()
        self._remainder_dim = self.dim - self._unit_dim * strategy.count_matrices()

        if self.requires_id_pairs_table:
            self.register_buffer(
                "_id_pairs_table", torch.from_numpy(id_pairs_table()).detach().clone()
            )

        # Build our custom embedding matrices
        embed = {}
        if strategy.glyphs:
            embed["glyphs"] = MyEmbedding(
                nh.MAX_GLYPH + 1, self._dim(strategy.glyphs), use_index_select
            )
        if strategy.colors:
            embed["colors"] = MyEmbedding(16, self._dim(strategy.colors), use_index_select)
        if strategy.chars:
            embed["chars"] = MyEmbedding(256, self._dim(strategy.chars), use_index_select)
        if strategy.specials:
            embed["specials"] = MyEmbedding(256, self._dim(strategy.specials), use_index_select)
        if strategy.groups:
            num_groups = self.id_pairs_table.select(1, 1).max().item() + 1
            embed["groups"] = MyEmbedding(num_groups, self._dim(strategy.groups), use_index_select)
        if strategy.subgroup_ids:
            num_subgroup_ids = self.id_pairs_table.select(1, 0).max().item() + 1
            embed["subgroup_ids"] = MyEmbedding(
                num_subgroup_ids, self._dim(strategy.subgroup_ids), use_index_select
            )

        self.embeddings = nn.ModuleDict(embed)
        self.targets = list(embed.keys())
        self.GlyphTuple = namedtuple("GlyphTuple", self.targets)

        if strategy.do_linear_layer and strategy.count_matrices() > 1:
            self.linear = nn.Linear(strategy.count_matrices() * self.dim, self.dim)

    def _dim(self, units):
        """Decide the embedding size for a single matrix.  If using a linear layer
        at the end this is always the embedding dimension, otherwise it is a
        fraction of the embedding dim"""
        if self.strategy.do_linear_layer:
            return self.dim
        else:
            dim = units * self._unit_dim + self._remainder_dim
            self._remainder_dim = 0
            return dim

    @property
    def requires_id_pairs_table(self):
        return self.strategy.groups or self.strategy.subgroup_ids

    @property
    def id_pairs_table(self):
        return self._id_pairs_table

    def prepare_input(self, inputs):
        """Take the inputs to the network as dictionary and return a namedtuple
        of the input/index tensors to be embedded (GlyphTuple)"""
        embeddable_data = {}
        # Only flatten the data we want
        for key, value in inputs.items():
            if key in self.embeddings:
                # -- [ T x B x ...] -> [ B' x ... ]
                embeddable_data[key] = torch.flatten(value, 0, 1).long()

        # add our group id and subgroup id if we want them
        if self.requires_id_pairs_table:
            ids, groups = self.glyphs_to_idgroup(inputs["glyphs"])
            embeddable_data["groups"] = groups
            embeddable_data["subgroup_ids"] = ids

        # convert embeddable_data to a named tuple
        return self.GlyphTuple(**embeddable_data)

    def forward(self, data_tuple):
        """Output the embdedded tuple prepared in in prepare input. This will be
        a GlyphTuple."""
        embs = []
        for field, data in zip(self.targets, data_tuple):
            embs.append(self.embeddings[field](data))
        if len(embs) == 1:
            return embs[0]

        embedded = torch.cat(embs, dim=-1)
        if self.strategy.do_linear_layer:
            embedded = self.linear(embedded)
        return embedded

    def glyphs_to_idgroup(self, glyphs):
        _, _, *shape = glyphs.shape
        ids_groups = self.id_pairs_table.index_select(0, glyphs.view(-1).long())
        ids = ids_groups.select(1, 0).view(-1, *shape).long()
        groups = ids_groups.select(1, 1).view(-1, *shape).long()
        return (ids, groups)


def make_field_glyph_emb(flags):
    field = flags.model.field

    return GlyphEmbedding(field.glyph_type, field.emb_dim, flags.model.use_index_select)


def make_inv_glyph_emb(flags, emb_dim):
    assert flags.model.inv.glyph_type in ("group_id", "full")

    return GlyphEmbedding(flags.model.inv.glyph_type, emb_dim, flags.model.use_index_select)
