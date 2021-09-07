import mmle.nn as mnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from nle.nethack import DUNGEON_SHAPE


class Crop(nn.Module):
    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        self.register_buffer(
            "width_grid",
            self._step_to_range(2 / (self.width - 1), self.width_target)[None, :]
            .expand(self.height_target, -1)
            .detach()
            .clone(),
        )
        self.register_buffer(
            "height_grid",
            self._step_to_range(2 / (self.height - 1), height_target)[:, None]
            .expand(-1, self.width_target)
            .detach()
            .clone(),
        )

    def _step_to_range(self, step, num_steps):
        return torch.tensor([step * (i - num_steps // 2) for i in range(num_steps)])

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.

        Args:
           inputs [B x H x W] or [B x H x W x C]
           coordinates [B x 2] x,y coordinates

        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height, "expected %d but found %d" % (
            self.height,
            inputs.shape[1],
        )
        assert inputs.shape[2] == self.width, "expected %d but found %d" % (
            self.width,
            inputs.shape[2],
        )

        permute_results = False
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        else:
            permute_results = True
            inputs = inputs.permute(0, 2, 3, 1)
        inputs = inputs.float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        crop = torch.round(F.grid_sample(inputs, grid, align_corners=True)).squeeze(1).long()

        if permute_results:
            # [B x C x H x W] -> [B x H x W x C]
            crop = crop.permute(0, 2, 3, 1)

        return crop


class FieldCnn(nn.Module):
    def __init__(self, emb_dim, height, width, cnn_depth, mid_channel, last_channel):
        super().__init__()

        if cnn_depth > 1:
            self._convs = nn.ModuleList([mnn.Conv(emb_dim, mid_channel, bn=False, activ="elu")])
            for _ in range(cnn_depth - 2):
                self._convs.append(mnn.Conv(mid_channel, mid_channel, bn=False, activ="elu"))
            self._convs.append(mnn.Conv(mid_channel, last_channel, bn=False, activ="elu"))
        elif cnn_depth == 1:
            self._convs = nn.ModuleList([mnn.Conv(emb_dim, last_channel, bn=False, activ="elu")])
        else:
            raise ValueError(f"cnn_depth == {cnn_depth}")

    def forward(self, x):
        """
        (B, E, w, h) -> (B, H)
        """

        for conv in self._convs:
            x = conv(x)  # (B, H', w, h)

        return x.flatten(1)  # (B, H=H'*h*w)


def _make_field_model(flags, field_model_flags, height, width):
    field = flags.model.field
    cnn = field_model_flags.cnn

    model_kind = field_model_flags.model
    if model_kind == "none":
        return None, 0
    elif model_kind == "cnn":
        return (
            FieldCnn(field.emb_dim, height, width, cnn.depth, cnn.mid_channel, cnn.last_channel),
            height * width * cnn.last_channel,
        )
    else:
        raise ValueError(f"model_kind == {model_kind}")


def make_field_models(flags):
    field = flags.model.field

    full_model, hidden1 = _make_field_model(flags, field.full, DUNGEON_SHAPE[0], DUNGEON_SHAPE[1])
    crop_model, hidden2 = _make_field_model(flags, field.crop, field.crop.dim, field.crop.dim)
    crop = Crop(DUNGEON_SHAPE[0], DUNGEON_SHAPE[1], field.crop.dim, field.crop.dim)

    return full_model, crop_model, crop, hidden1 + hidden2
