# NOTICE: This implementation of DISK is based on that of kornia see
# https://github.com/kornia/kornia/blob/a041d9255f459a0034c75b20fb9ab524d7ae3983/kornia/feature/disk/disk.py

# Adapted by Arthur Elskens

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from kornia.color import grayscale_to_rgb
from kornia.core import Device
from torch import nn

from .unet import Unet
from .utils import (
    Keypoints,
    heatmap_to_keypoints,
    numpy_image_to_torch,
    remove_batch_dimension,
    smart_loader,
)

MODELS_PATH = os.path.join(Path(os.path.abspath(__file__)).parents[2], "models")


class FlexibleDISK(nn.Module):
    def __init__(
        self,
        desc_dim: int = 128,
        unet: Optional[nn.Module] = None,
        max_num_keypoints: Optional[int] = None,
        nms_window_size: int = 5,
        detection_threshold: float = 0.0,
        pad_if_not_divisible: bool = True,
    ) -> None:
        super().__init__()

        self.desc_dim = desc_dim

        if unet is None:
            unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, desc_dim + 1])

        self.unet = unet

        self.max_num_keypoints = max_num_keypoints
        self.nms_window_size = nms_window_size
        self.detection_threshold = detection_threshold
        self.pad_if_not_divisible = pad_if_not_divisible

    def forward(
        self,
        img: torch.Tensor,
        keypoints: Optional[torch.Tensor] = None,
        detection: bool = False,
        description: bool = False,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Detects features in an image, returning keypoint locations, descriptors and detection scores.

        :param img: The image from which the keypoints will be described. Shape `(B, 3, H, W)`.
        :type img: torch.Tensor
        :param keypoints: The keypoints to describe, defaults to None.
        :type keypoints: Optional[torch.Tensor], optional
        :param detection: Whether to do the detection or not, defaults to False.
        :type detection: bool, optional
        :param description: Whether to do the description or not, defaults to False.
        :type description: bool, optional
        :return keypoints: The detected keypoints.
        :rtype: Optional[torch.Tensor]
        :return scores: The score associated to the keypoints.
        :rtype: Optional[torch.Tensor]
        :return descriptors: The associated descriptors.
        :rtype: Optional[torch.Tensor]
        """

        features = Keypoints(keypoints, torch.ones((len(keypoints), 1))) if keypoints is not None else None

        assert isinstance(img, torch.Tensor), "The given input image is not a Tensor."

        if img.shape[1] == 1:
            img = grayscale_to_rgb(img)

        if self.pad_if_not_divisible:
            h, w = img.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            img = torch.nn.functional.pad(img, (0, pd_w, 0, pd_h), value=0.0)

        heatmaps, descriptors = self.heatmap_and_dense_descriptors(img)
        if self.pad_if_not_divisible:
            heatmaps = heatmaps[..., :h, :w]
            descriptors = descriptors[..., :h, :w]

        if detection:
            # List due to batch dimension -> take the first because just one in inference
            features = heatmap_to_keypoints(
                heatmaps,
                n=self.max_num_keypoints,
                window_size=self.nms_window_size,
                score_threshold=self.detection_threshold,
            )[0]

        if description:
            assert features is not None
            # Again due to batch dimension -> take the first because just one in inference
            tmp = features.merge_with_descriptors(descriptors[0])
            keypoints, scores, descriptors = tmp.keypoints, tmp.detection_scores, tmp.descriptors
        else:
            keypoints = features.xys.to(descriptors.dtype) if features else None
            scores = features.detection_logp if features else None

        return keypoints, scores, descriptors

    def heatmap_and_dense_descriptors(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the heatmap and the dense descriptors.

        :param images: The image to detect features in. Shape `(B, 3, H, W)`.
        :type images: torch.Tensor
        :return: A tuple of dense detection scores and descriptors. Shapes are `(B, 1, H, W)` and
        `(B, D, H, W)`, where `D` is the descriptor dimension.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        unet_output = self.unet(images)

        if unet_output.shape[1] != self.desc_dim + 1:
            raise ValueError(
                f"U-Net output has {unet_output.shape[1]} channels, but expected self.desc_dim={self.desc_dim} + 1."
            )

        descriptors = unet_output[:, : self.desc_dim]
        heatmaps = unet_output[:, self.desc_dim :]

        return heatmaps, descriptors

    def load(self, path: str = MODELS_PATH, device: Device = torch.device("cpu")) -> None:
        loader = smart_loader(path)
        model = loader(path if "http" in path else f"{path}/{self.__class__.__name__}.pth", map_location=device)

        self.load_state_dict(model["extractor"])

    @torch.no_grad()
    def detect(self, img: np.ndarray) -> Sequence[cv2.KeyPoint]:
        """Implementation of OpenCV's Feature2D `detect()` method.

        :param img: The image from which the keypoints will be detected.
        :type img: np.ndarray
        :return: The detected keypoints.
        :rtype: Sequence[cv2.KeyPoint]
        """

        tf_img = numpy_image_to_torch(img)
        keypoints, scores, _ = self.forward(tf_img, detection=True)

        assert keypoints is not None and scores is not None
        tmp = []
        for kp, score in zip(remove_batch_dimension(keypoints), remove_batch_dimension(scores)):
            tmp.append(cv2.KeyPoint(x=kp[0].item(), y=kp[1].item(), size=1.0, response=score.item()))

        return tuple(tmp)

    @torch.no_grad()
    def compute(self, img: np.ndarray, keypoints: Sequence[cv2.KeyPoint]) -> tuple[Sequence[cv2.KeyPoint], np.ndarray]:
        """Implementation of OpenCV's Feature2D `detect()` method.

        :param img: The image from which the keypoints will be described.
        :type img: np.ndarray
        :param keypoints: The keypoints to describe.
        :type keypoints: Sequence[cv2.KeyPoint]
        :return keypoints: The detected keypoints.
        :rtype: Sequence[cv2.KeyPoint]
        :return descriptors: The associated descriptors.
        :rtype: np.ndarray
        """

        tf_img = numpy_image_to_torch(img)
        # Expected input: torch.tensor(shape=(n_kp, 2))
        tf_keypoints = torch.stack([torch.tensor(kp.pt) for kp in keypoints])

        new_keypoints, scores, descriptors = self.forward(tf_img, tf_keypoints, description=True)

        assert new_keypoints is not None and scores is not None and descriptors is not None

        tmp = []
        for kp, score in zip(remove_batch_dimension(new_keypoints), remove_batch_dimension(scores)):
            tmp.append(cv2.KeyPoint(x=kp[0].item(), y=kp[1].item(), size=1.0, response=score.item()))

        return tuple(tmp), remove_batch_dimension(descriptors).numpy(force=True)

    @torch.no_grad()
    def detectAndCompute(self, img: np.ndarray) -> tuple[Sequence[cv2.KeyPoint], np.ndarray]:
        """Implementation of OpenCV's Feature2D `detect()` method.

        :param img: The image from which the keypoints will be described and described.
        :type img: np.ndarray
        :return keypoints: The detected keypoints.
        :rtype: Sequence[cv2.KeyPoint]
        :return descriptors: The associated descriptors.
        :rtype: np.ndarray
        """

        tf_img = numpy_image_to_torch(img)
        keypoints, scores, descriptors = self.forward(tf_img, detection=True, description=True)

        assert keypoints is not None and scores is not None and descriptors is not None
        tmp = []
        for kp, score in zip(remove_batch_dimension(keypoints), remove_batch_dimension(scores)):
            tmp.append(cv2.KeyPoint(x=kp[0].item(), y=kp[1].item(), size=1.0, response=score.item()))

        return tuple(tmp), remove_batch_dimension(descriptors).numpy(force=True)
