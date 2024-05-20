# NOTICE: Utils for DISK taken from kornia.

# Adapted by Arthur Elskens

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from kornia.core import Device
from torchvision import transforms


@dataclass
class DISKFeatures:
    r"""A data structure holding DISK keypoints, descriptors and detection scores for an image. Since DISK detects a
    varying number of keypoints per image, `DISKFeatures` is not batched.

    Args:
        keypoints: Tensor of shape :math:`(N, 2)`, where :math:`N` is the number of keypoints.
        descriptors: Tensor of shape :math:`(N, D)`, where :math:`D` is the descriptor dimension.
        detection_scores: Tensor of shape :math:`(N,)` where the detection score can be interpreted as
                          the log-probability of keeping a keypoint after it has been proposed (see the paper
                          section *Method → Feature distribution* for details).
    """

    keypoints: torch.Tensor
    descriptors: torch.Tensor
    detection_scores: torch.Tensor

    @property
    def n(self) -> int:
        return self.keypoints.shape[0]

    @property
    def device(self) -> Device:
        return self.keypoints.device

    @property
    def x(self) -> torch.Tensor:
        """Accesses the x coordinates of keypoints (along image width)."""
        return self.keypoints[:, 0]

    @property
    def y(self) -> torch.Tensor:
        """Accesses the y coordinates of keypoints (along image height)."""
        return self.keypoints[:, 1]

    def to(self, *args: Any, **kwargs: Any) -> DISKFeatures:
        """Calls :func:`torch.Tensor.to` on each tensor to move the keypoints, descriptors and detection scores to
        the specified device and/or data type.

        Args:
            *args: Arguments passed to :func:`torch.Tensor.to`.
            **kwargs: Keyword arguments passed to :func:`torch.Tensor.to`.

        Returns:
            A new DISKFeatures object with tensors of appropriate type and location.
        """
        return DISKFeatures(
            self.keypoints.to(*args, **kwargs),
            self.descriptors.to(*args, **kwargs),
            self.detection_scores.to(*args, **kwargs),
        )


@dataclass
class Keypoints:
    """A temporary struct used to store keypoint detections and their log-probabilities.

    After construction, merge_with_descriptors is used to select corresponding descriptors from unet output.
    """

    xys: torch.Tensor
    detection_logp: torch.Tensor

    def _remove_out_of_bounds_xys(self, dims: tuple[int, int]) -> None:
        """Remove out-of-bounds xy and update the scores as well.

        :param dims: The maximum dimensions, given as (x, y).
        :type dims: tuple[int, int]
        """

        xys = []
        scores = []
        for xy, score in zip(self.xys, self.detection_logp):
            if xy[0] <= dims[0] and xy[1] <= dims[1]:
                xys.append(xy)
                scores.append(score)

        self.xys = torch.stack(xys)
        self.detection_logp = torch.stack(scores)

    def merge_with_descriptors(self, descriptors: torch.Tensor) -> DISKFeatures:
        """Select descriptors from a dense `descriptors` tensor, at locations given by `self.xys`"""

        dtype = descriptors.dtype

        # Remove out-of-bounds xy
        self._remove_out_of_bounds_xys((descriptors.shape[-1], descriptors.shape[-2]))

        x, y = self.xys.T

        desc = descriptors[:, torch.round(y).to(torch.int), torch.round(x).to(torch.int)].T
        desc = F.normalize(desc, dim=-1)

        return DISKFeatures(self.xys.to(dtype), desc, self.detection_logp)


def nms(signal: torch.Tensor, window_size: int = 5, cutoff: float = 0.0) -> torch.Tensor:
    if window_size % 2 != 1:
        raise ValueError(f"window_size has to be odd, got {window_size}")

    _, ixs = F.max_pool2d(signal, kernel_size=window_size, stride=1, padding=window_size // 2, return_indices=True)

    h, w = signal.shape[1:]
    coords = torch.arange(h * w, device=signal.device).reshape(1, h, w)
    nms = ixs == coords

    if cutoff is None:
        return nms
    else:
        return nms & (signal > cutoff)


def heatmap_to_keypoints(
    heatmap: torch.Tensor, n: Optional[int] = None, window_size: int = 5, score_threshold: float = 0.0
) -> list[Keypoints]:
    """Inference-time nms-based detection protocol."""
    heatmap = heatmap.squeeze(1)
    nmsed = nms(heatmap, window_size=window_size, cutoff=score_threshold)

    keypoints = []
    for b in range(heatmap.shape[0]):
        yx = nmsed[b].nonzero(as_tuple=False)
        detection_logp = heatmap[b][nmsed[b]]
        xy = yx.flip((1,))

        if n is not None:
            n_ = min(n + 1, detection_logp.numel())
            # torch.kthvalue picks in ascending order and we want to pick in
            # descending order, so we pick n-th smallest among -logp to get
            # -threshold
            minus_threshold, _indices = torch.kthvalue(-detection_logp, n_)
            mask = detection_logp > -minus_threshold

            xy = xy[mask]
            detection_logp = detection_logp[mask]

            # it may be that due to numerical saturation on the threshold we have
            # more than n keypoints, so we need to clip them
            xy = xy[:n]
            detection_logp = detection_logp[:n]

        keypoints.append(Keypoints(xy, detection_logp))

    return keypoints


def numpy_image_to_torch(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image to torch.

    :param img: The input numpy image to convert to torch.
    :type img: np.ndarray
    :return: The converted torch image.
    :rtype: torch.Tensor
    """

    numpy_to_torch = transforms.ToTensor()

    if img.dtype == float and img.dtype != np.float32:
        img = img.astype(np.float32)

    return numpy_to_torch(img).unsqueeze(0)


def smart_loader(path: str) -> Callable:
    """Determine which state_dict loader to use.

    :param path: The path to the model's .pth file.
    :type path: str
    :return: The appropriate loader to use.
    :rtype: Callable
    """

    if "http" in path:
        return torch.hub.load_state_dict_from_url

    return torch.load


def remove_batch_dimension(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(0)
