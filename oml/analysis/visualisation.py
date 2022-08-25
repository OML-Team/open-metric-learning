from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.images.images import imread_cv2

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (120, 120, 120)

TColor = Tuple[int, int, int]


def draw_bbox(im: np.ndarray, bbox: torch.Tensor, color: TColor) -> np.ndarray:
    """
    Draws a single bounding box on the image.
    If the elements of the bbox are NaNs, we will draw bbox around the whole image.

    Args:
        im: image
        bbox: single bounding in the format of [x1, y1, x2, y2]
        color: tuple of 3 ints
    """
    im_ret = im.copy()
    if not torch.isnan(bbox[0]):
        x1, y1, x2, y2 = list(map(int, bbox))
    else:
        x1, y1, x2, y2 = 0, 0, im_ret.shape[1], im_ret.shape[0]

    im_ret = cv2.rectangle(im_ret, (x1, y1), (x2, y2), thickness=5, color=color)

    return im_ret


class RetrievalVisualizer:
    def __init__(
        self,
        query_paths: List[str],
        query_labels: torch.Tensor,
        gallery_paths: List[str],
        gallery_labels: torch.Tensor,
        dist_matrix: torch.Tensor,
        mask_to_ignore: torch.Tensor,
        mask_gt: torch.Tensor,
        query_bboxes: torch.Tensor,
        gallery_bboxes: torch.Tensor,
    ):
        """
        This class allows you to visualize the searching results for the desired queries and
        highlight bad and good predictions with the different colors.

        Args:
            query_paths: Q paths to queries images
            query_labels: Q labels of query images
            gallery_paths: G paths to gallery images
            gallery_labels: G labels of gallery images
            dist_matrix: (Q x G) matrix of the distances between queries and galleries
            mask_to_ignore: (Q x G) boolean matrix. If (i,j) is True we will ignore j-th gallery for i-th query.
            mask_gt: (Q x G) boolean matrix. The (i,j) element indicates if j-th gallery is the correct output for the i-th query.
            query_bboxes: (Q x 4) matrix of the bboxes in the format of [x1, y1, x2, y2]. Use torch("nan") if you have no bboxes for the sample.
            gallery_bboxes: (G x 4) matrix of the bboxes in the format of [x1, y1, x2, y2]. Use torch("nan") if you have no bboxes for the sample.

        """
        self.query_paths = query_paths
        self.query_labels = query_labels

        self.gallery_paths = gallery_paths
        self.gallery_labels = gallery_labels

        self.dist_matrix = dist_matrix
        self.mask_gt = mask_gt

        self.dist_matrix[mask_to_ignore] = float("inf")

        self.query_bboxes = query_bboxes
        self.gallery_bboxes = gallery_bboxes

    @classmethod
    def from_embeddings_metric(cls, emb: EmbeddingMetrics) -> "RetrievalVisualizer":
        """
        In some cases, you may prefer to instantiate Visualizer from
        >>> EmbeddingMetrics
        which you usually have after the validation procedure since it
        contains the information needed for visualization.

        Args:
            emb: EmbeddingMetrics object

        """
        is_query = emb.acc.storage[emb.is_query_key]
        is_gallery = emb.acc.storage[emb.is_gallery_key]

        query_paths = np.array(emb.acc.storage["paths"])[is_query]
        gallery_paths = np.array(emb.acc.storage["paths"])[is_gallery]

        query_labels = emb.acc.storage[emb.labels_key][is_query]  # type: ignore
        gallery_labels = emb.acc.storage[emb.labels_key][is_gallery]  # type: ignore

        bboxes = list(zip(emb.acc.storage["x1"], emb.acc.storage["y1"], emb.acc.storage["x2"], emb.acc.storage["y2"]))

        query_bboxes = torch.tensor(bboxes)[is_query]
        gallery_bboxes = torch.tensor(bboxes)[is_gallery]

        return RetrievalVisualizer(
            query_paths=query_paths,
            query_labels=query_labels,
            gallery_paths=gallery_paths,
            gallery_labels=gallery_labels,
            dist_matrix=emb.distance_matrix,
            mask_to_ignore=emb.mask_to_ignore,
            mask_gt=emb.mask_gt,
            query_bboxes=query_bboxes,
            gallery_bboxes=gallery_bboxes,
        )

    def visualise(
        self,
        query_idx: int,
        top_k: int,
        skip_no_errors: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Visualize the predictions for the query with the index <query_idx>.

        Args:
            query_idx: Index of the query
            top_k: Amount of the predictions to show
            skip_no_errors: We skip the current query if we have no errors among first top_k predictions

        """
        ids = torch.argsort(self.dist_matrix[query_idx])[:top_k]

        if skip_no_errors and torch.all(self.mask_gt[query_idx, ids]):
            print(f"No errors for {query_idx}")
            return None

        n_gt = self.mask_gt[query_idx].sum()
        ngt_show = 2
        fig = plt.figure(figsize=(30, 15))

        plt.subplot(1, top_k + 1 + ngt_show, 1)

        img = self.get_img_with_bbox(self.query_paths[query_idx], self.query_bboxes[query_idx], BLUE)
        print("Q  ", self.query_paths[query_idx])
        plt.imshow(img)
        plt.title(f"Query, #gt = {n_gt}")
        plt.axis("off")

        for i, idx in enumerate(ids):
            color = GREEN if self.mask_gt[query_idx, idx] else RED
            print("G", i, self.gallery_paths[idx])
            plt.subplot(1, top_k + ngt_show + 1, i + 2)
            img = self.get_img_with_bbox(self.gallery_paths[idx], self.gallery_bboxes[idx], color)

            plt.title(f"{i} - {round(self.dist_matrix[query_idx, idx].item(), 3)}")
            plt.imshow(img)
            plt.axis("off")

        gt_ids = self.mask_gt[query_idx].nonzero(as_tuple=True)[0][:ngt_show]

        for i, gt_idx in enumerate(gt_ids):
            plt.subplot(1, top_k + ngt_show + 1, i + top_k + 2)
            img = self.get_img_with_bbox(self.gallery_paths[gt_idx], self.gallery_bboxes[gt_idx], GRAY)
            plt.title("GT " + str(round(self.dist_matrix[query_idx, gt_idx].item(), 3)))
            plt.imshow(img)
            plt.axis("off")

        plt.show()
        return fig

    @staticmethod
    def get_img_with_bbox(im_name: str, bbox: torch.Tensor, color: TColor) -> np.ndarray:
        """
        Reads the image by its name and draws bbox on it.

        Args:
            im_name: Image path
            bbox: Single bounding box in the format of [x1, y1, x2, y2]. It may also be a list of 4 torch("nan").
            color: Tuple of 3 ints from 0 to 255
        """
        img = imread_cv2(im_name)
        img = draw_bbox(img, bbox, color)
        return img


__all__ = ["RED", "GREEN", "BLUE", "GRAY", "TColor", "draw_bbox", "RetrievalVisualizer"]
