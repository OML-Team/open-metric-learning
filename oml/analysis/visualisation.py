from typing import List, Tuple

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

    def visualise(self, query_idx: int, top_k: int, skip_no_errors: bool = False) -> None:
        ids = torch.argsort(self.dist_matrix[query_idx])[:top_k]

        if skip_no_errors and torch.all(self.mask_gt[query_idx, ids]):
            print(f"No errors for {query_idx}")
            return

        n_gt = self.mask_gt[query_idx].sum()
        ngt_show = 2
        plt.figure(figsize=(30, 15))

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

    @staticmethod
    def get_img_with_bbox(im_name: str, bbox: Tuple[int, int, int, int], color: TColor) -> np.ndarray:
        img = imread_cv2(im_name)
        img = draw_bbox(img, bbox, color)
        return img


__all__ = ["RED", "GREEN", "BLUE", "GRAY", "TColor", "draw_bbox", "RetrievalVisualizer"]
