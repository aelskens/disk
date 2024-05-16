import pickle

import numpy as np
import pytest
import torch

from src.disk import FlexibleDISK, numpy_image_to_torch, remove_batch_dimension


def pickle_load(path):
    with open(path, "rb") as fp:
        out = pickle.load(fp)

    return out


# .npy of two different images: (i) a Whole-Slide Image and (ii) a building image
wsi = np.load("./tests/wsi.npy")
building = np.load("./tests/building.npy")

# The pickled output of kornia's DISK implementation for both image
# with a `max_num_keypoints=20000`and `resize=None`
wsi_expected = pickle_load("./tests/disk_wsi.pickle")
building_expected = pickle_load("./tests/disk_building.pickle")


@pytest.mark.parametrize(
    "img, expected",
    [(wsi, wsi_expected), (building, building_expected)],
)
def test_coherence_with_disk(img, expected) -> None:
    """Unit test to validate that the results of FlexibleSuperPoint correspond to that of the original implementation."""

    def _compare(list1, list2):
        for i, j in zip(list1, list2):
            if not torch.all(i == j):
                return False

        return True

    model = FlexibleDISK(max_num_keypoints=20000).eval()
    model.load("https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth")

    assert _compare(
        list(model(numpy_image_to_torch(img), detection=True, description=True)),
        [i for i in (expected[0].keypoints, expected[0].detection_scores, expected[0].descriptors)],
    )


@pytest.mark.parametrize(
    "img, mode, expected",
    [(wsi, "complete", wsi_expected), (wsi, "split", wsi_expected)],
)
def test_opencv_interface(img, mode, expected) -> None:
    """Unit test to validate that the OPENCV interface yields the same results as the original implementation."""

    model = FlexibleDISK(max_num_keypoints=20000).eval()
    model.load("https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth")

    match mode:
        case "complete":
            _, desc = model.detectAndCompute(img)
        case "split":
            kp = model.detect(img)
            _, desc = model.compute(img, kp)

    assert np.all(remove_batch_dimension(expected[0].descriptors).detach().numpy() == desc)
