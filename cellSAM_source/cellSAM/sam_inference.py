from typing import Tuple
import copy

import numpy as np
import warnings
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
from torchvision import tv_tensors
from sklearn.cluster import KMeans

import torchvision.transforms.v2 as T
from .AnchorDETR.models import build_inference_model
from .AnchorDETR import transforms as anchorT
from .AnchorDETR.util.box_ops import box_cxcywh_to_xyxy
from segment_anything import (
    sam_model_registry,
)
from scipy import ndimage


def keep_largest_object(img: np.ndarray) -> np.ndarray:
    """
    Keep only the largest object in the binary image (np.array).
    """
    img_array = img
    label_image, _ = ndimage.label(img_array)
    label_histogram = np.bincount(label_image.ravel())
    label_histogram[0] = 0  # Clear the background label
    largest_object_label = label_histogram.argmax()
    cleaned_array = np.where(label_image == largest_object_label, img_array.max(), 0)

    return cleaned_array


# from sam repo
class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(
            oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CellfinderAnchorDetr(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        # Set all parameters from modelconfig.yaml
        args.enc_layers = 6
        args.dec_layers = 6
        args.dim_feedforward = 1024
        args.hidden_dim = 256
        args.dropout = 0.0
        args.nheads = 8
        args.num_query_position = 3500
        args.num_query_pattern = 1
        args.spatial_prior = "learned"
        args.attention_type = "RCDA"
        args.num_feature_levels = 1
        args.device = "cuda"
        args.num_classes = 2
        args.prior_mode = getattr(args, "prior_mode", "strict")
        
        # Additional required parameters
        args.in_channels = 768
        args.backbone = "SAM"
        args.only_neck = False
        args.freeze_backbone = False
        args.sam_vit = "vit_b"

        if not hasattr(self, "decode_head"):
            self.decode_head, self.postprocessors = build_inference_model(args)

    def forward(
        self,
        features=None,
        candidate_points=None,
        candidate_valid_mask=None,
        prior_mode=None,
        apply_candidate_mask=True,
    ):
        outputs = self.decode_head(
            features,
            candidate_points=candidate_points,
            candidate_valid_mask=candidate_valid_mask,
            prior_mode=prior_mode or self.args.prior_mode,
            apply_candidate_mask=apply_candidate_mask,
        )
        return outputs

    @torch.no_grad()
    def forward_inference(
        self,
        imgs,
        viz=False,
        candidate_points=None,
        candidate_valid_mask=None,
        prior_mode=None,
        apply_candidate_mask=True,
        return_raw_outputs=False,
    ):
        outputs = self.decode_head(
            imgs,
            candidate_points=candidate_points,
            candidate_valid_mask=candidate_valid_mask,
            prior_mode=prior_mode or self.args.prior_mode,
            apply_candidate_mask=apply_candidate_mask,
        )
        if return_raw_outputs:
            return outputs

        orig_target_sizes = [torch.tensor(img.shape[-2:]) for img in imgs]
        orig_target_sizes = torch.stack(orig_target_sizes, dim=0)
        orig_target_sizes = orig_target_sizes.to(imgs.device)

        res = self.postprocessors["bbox"](outputs, orig_target_sizes)
        return res


class CellSAM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = sam_model_registry["vit_b"]()
        self.mask_threshold = 0.4
        self.iou_threshold = 0.5
        self.bbox_threshold = 0.4
        self.sam_transform = ResizeLongestSide(1024)

        config = Namespace(**config)
        self.cellfinder = CellfinderAnchorDetr(config)

        self.adv_mode = True
        self.model_cp = copy.deepcopy(self.model)

        # Transforms
        self.normalize = T.Compose([
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_transforms(self, imgs):
        imgs = [tv_tensors.Image(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)

        return imgs

    def sam_preprocess(self, x: torch.Tensor, return_paddings=False, div_255=False):
        """Normalize pixel values and pad to a square input."""
        mean = self.model.pixel_mean
        std = self.model.pixel_std
        if div_255:
            mean = mean / 255
            std = std / 255
        x = (x - mean) / std

        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        if return_paddings:
            return x, (padh, padw)
        else:
            return x

    def sam_bbox_preprocessing(self, imgs, percentile=True):
        imgs = [T.Resize((1024, 1024))(img) for img in imgs]
        x = [torchvision.transforms.ToPILImage()(img) for img in imgs]
        x = [np.array(img) for img in x]
        x = [self.sam_transform.apply_image(img) for img in x]
        device = next(self.parameters()).device
        x = [torch.from_numpy(img).permute(2, 0, 1).contiguous().to(device) for img in x]
        x = [self.sam_preprocess(img, return_paddings=True) for img in x]
        imgs, paddings = zip(*x)
        if percentile:
            imgs = [anchorT.PercentileThreshold()(img.cpu()) for img in imgs]
        imgs = [torch.Tensor(img) for img in imgs]

        if self.adv_mode:
            imgs = [self.normalize(img) for img in imgs]
            imgs = [anchorT.Standardize()(img) for img in imgs]
            imgs = [anchorT.ToRGB()(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        device = next(self.parameters()).device
        imgs = imgs.to(device)

        return imgs

    def sam_preprocess_pad(self, x: torch.Tensor, return_paddings=False):
        h, w = x.shape[-2:]
        padh = self.model.image_encoder.img_size - h
        padw = self.model.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        if return_paddings:
            return x, (padh, padw)
        else:
            return x

    def forward(self, x, return_preprocessed=False, *args, **kwargs):
        device = next(self.parameters()).device
        x = x.to(device)
        x = [self.sam_preprocess(img, return_paddings=True, div_255=True) for img in x]
        x, paddings = zip(*x)
        preprocessed_img = torch.stack(x, dim=0)

        if self.adv_mode:
            x = self.model_cp.image_encoder(preprocessed_img)
        else:
            x = self.model.image_encoder(preprocessed_img)

        if return_preprocessed:
            return x, preprocessed_img, paddings
        else:
            return x

    def prep_2(self, imgs, percentile=True):
        imgs = [T.Resize((1024, 1024))(img) for img in imgs]
        imgs = [self.sam_preprocess_pad(img, return_paddings=True) for img in imgs]
        imgs, paddings = zip(*imgs)

        if percentile:
            imgs = [anchorT.PercentileThreshold()(img.cpu()) for img in imgs]
        imgs = [torch.Tensor(img) for img in imgs]
        if self.adv_mode:
            imgs = [self.normalize(img) for img in imgs]
            imgs = [anchorT.Standardize()(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)

        return imgs, paddings

    def _prepare_candidate_prior_batch(
        self,
        candidate_points_per_image,
        candidate_valid_masks,
        batch_size,
        max_queries,
        device,
    ):
        if candidate_points_per_image is None:
            return None, None

        def _pad_or_truncate(points_tensor, valid_mask_tensor):
            num_points = points_tensor.shape[0]
            if num_points < max_queries:
                pad_count = max_queries - num_points
                pad_points = torch.full(
                    (pad_count, 2),
                    0.5,
                    dtype=torch.float32,
                    device=device,
                )
                pad_mask = torch.zeros((pad_count,), dtype=torch.bool, device=device)
                points_tensor = torch.cat([points_tensor, pad_points], dim=0)
                valid_mask_tensor = torch.cat([valid_mask_tensor, pad_mask], dim=0)
            elif num_points > max_queries:
                points_tensor = points_tensor[:max_queries]
                valid_mask_tensor = valid_mask_tensor[:max_queries]

            return torch.clamp(points_tensor, 0.0, 1.0), valid_mask_tensor

        if torch.is_tensor(candidate_points_per_image):
            candidate_points = candidate_points_per_image.to(device=device, dtype=torch.float32)
            if candidate_points.dim() == 2:
                if batch_size != 1:
                    raise ValueError(
                        "2D candidate_points_per_image is only valid for batch_size=1"
                    )
                candidate_points = candidate_points.unsqueeze(0)
            if candidate_points.dim() != 3 or candidate_points.shape[0] != batch_size:
                raise ValueError(
                    "candidate_points_per_image must have shape [B, Q, 2], "
                    f"got {tuple(candidate_points.shape)}"
                )
            if candidate_points.shape[-1] != 2:
                raise ValueError(
                    "candidate_points_per_image last dim must be 2, "
                    f"got {candidate_points.shape[-1]}"
                )

            if candidate_valid_masks is None:
                candidate_valid_mask = torch.ones(
                    candidate_points.shape[:2],
                    dtype=torch.bool,
                    device=device,
                )
            else:
                candidate_valid_mask = torch.as_tensor(
                    candidate_valid_masks,
                    device=device,
                    dtype=torch.bool,
                )
                if candidate_valid_mask.dim() == 1:
                    if batch_size != 1:
                        raise ValueError(
                            "1D candidate_valid_masks is only valid for batch_size=1"
                        )
                    candidate_valid_mask = candidate_valid_mask.unsqueeze(0)
                if candidate_valid_mask.shape != candidate_points.shape[:2]:
                    raise ValueError(
                        "candidate_valid_masks must match candidate_points_per_image shape [B, Q], "
                        f"got {tuple(candidate_valid_mask.shape)} vs {tuple(candidate_points.shape[:2])}"
                    )

            point_batches = []
            mask_batches = []
            for index in range(batch_size):
                points_tensor, valid_mask_tensor = _pad_or_truncate(
                    candidate_points[index],
                    candidate_valid_mask[index],
                )
                point_batches.append(points_tensor)
                mask_batches.append(valid_mask_tensor)
            return torch.stack(point_batches, dim=0), torch.stack(mask_batches, dim=0)

        if len(candidate_points_per_image) != batch_size:
            raise ValueError(
                "candidate_points_per_image length must match batch size, "
                f"got {len(candidate_points_per_image)} vs {batch_size}"
            )

        if candidate_valid_masks is None:
            candidate_valid_masks = [None] * batch_size
        elif len(candidate_valid_masks) != batch_size:
            raise ValueError(
                "candidate_valid_masks length must match batch size, "
                f"got {len(candidate_valid_masks)} vs {batch_size}"
            )

        point_batches = []
        mask_batches = []
        for points_item, mask_item in zip(candidate_points_per_image, candidate_valid_masks):
            points_tensor = torch.as_tensor(points_item, device=device, dtype=torch.float32)
            if points_tensor.numel() == 0:
                points_tensor = points_tensor.reshape(0, 2)
            if points_tensor.dim() != 2 or points_tensor.shape[-1] != 2:
                raise ValueError(
                    "Each candidate point array must have shape [Q, 2], "
                    f"got {tuple(points_tensor.shape)}"
                )

            if mask_item is None:
                valid_mask_tensor = torch.ones(
                    (points_tensor.shape[0],),
                    dtype=torch.bool,
                    device=device,
                )
            else:
                valid_mask_tensor = torch.as_tensor(
                    mask_item,
                    device=device,
                    dtype=torch.bool,
                ).reshape(-1)
                if valid_mask_tensor.shape[0] != points_tensor.shape[0]:
                    raise ValueError(
                        "Each candidate mask must match its point count, "
                        f"got {valid_mask_tensor.shape[0]} vs {points_tensor.shape[0]}"
                    )

            points_tensor, valid_mask_tensor = _pad_or_truncate(
                points_tensor,
                valid_mask_tensor,
            )
            point_batches.append(points_tensor)
            mask_batches.append(valid_mask_tensor)

        return torch.stack(point_batches, dim=0), torch.stack(mask_batches, dim=0)

    def _target_sizes_from_batch(self, imgs):
        target_sizes = [torch.tensor(img.shape[-2:]) for img in imgs]
        target_sizes = torch.stack(target_sizes, dim=0)
        return target_sizes.to(imgs.device)

    def _raw_outputs_to_query_results(self, outputs, target_sizes):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        scores = pred_logits.sigmoid()[..., 0]
        boxes = box_cxcywh_to_xyxy(pred_boxes)

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        effective_valid_mask = outputs.get("effective_candidate_valid_mask")
        query_results = []
        for batch_index in range(boxes.shape[0]):
            item = {
                "boxes": boxes[batch_index],
                "scores": scores[batch_index],
            }
            if effective_valid_mask is not None:
                item["effective_candidate_valid_mask"] = effective_valid_mask[batch_index]
            query_results.append(item)
        return query_results

    def _candidate_aligned_boxes_from_query_results(self, query_results):
        aligned_boxes = []
        for result in query_results:
            valid_mask = result.get("effective_candidate_valid_mask")
            if valid_mask is None:
                raise ValueError(
                    "candidate_aligned output requires effective_candidate_valid_mask "
                    "from prior-aware query outputs."
                )
            aligned_boxes.append(result["boxes"][valid_mask])
        return aligned_boxes

    @torch.no_grad()
    def generate_query_outputs(
        self,
        images,
        device=None,
        candidate_points_per_image=None,
        candidate_valid_masks=None,
        prior_mode=None,
        apply_candidate_mask=True,
    ):
        transformed_imgs_anchor = self.sam_bbox_preprocessing(images, percentile=False)
        candidate_points, candidate_valid_mask = self._prepare_candidate_prior_batch(
            candidate_points_per_image=candidate_points_per_image,
            candidate_valid_masks=candidate_valid_masks,
            batch_size=transformed_imgs_anchor.shape[0],
            max_queries=self.cellfinder.args.num_query_position,
            device=transformed_imgs_anchor.device,
        )
        raw_outputs = self.cellfinder.forward_inference(
            transformed_imgs_anchor,
            candidate_points=candidate_points,
            candidate_valid_mask=candidate_valid_mask,
            prior_mode=prior_mode,
            apply_candidate_mask=apply_candidate_mask,
            return_raw_outputs=True,
        )
        target_sizes = self._target_sizes_from_batch(transformed_imgs_anchor)
        return self._raw_outputs_to_query_results(raw_outputs, target_sizes)

    def _filter_boxes_by_scores(
        self,
        boxes_per_heatmap,
        pred_scores,
        score_filter_mode="dynamic",
        score_threshold=None,
    ):
        """Filter postprocessed boxes using the requested score policy."""
        score_filter_mode = score_filter_mode or "dynamic"
        if score_filter_mode == "none":
            return boxes_per_heatmap

        filtered_boxes = []
        for boxes, scores in zip(boxes_per_heatmap, pred_scores):
            score_data = scores.detach().cpu().numpy()
            if len(score_data) == 0:
                filtered_boxes.append(boxes)
                continue

            if score_filter_mode == "dynamic":
                threshold = self.bbox_threshold
                if len(score_data) > 1:
                    try:
                        kmeans = KMeans(n_clusters=2, random_state=42).fit(
                            score_data.reshape(-1, 1)
                        )
                        cluster_centers = kmeans.cluster_centers_
                        threshold_cluster = np.mean(cluster_centers)
                        threshold = 0.66 * self.bbox_threshold + 0.33 * threshold_cluster
                    except:
                        pass
            elif score_filter_mode == "fixed":
                threshold = (
                    self.bbox_threshold if score_threshold is None else score_threshold
                )
            else:
                raise ValueError(
                    "Unsupported score_filter_mode="
                    f"{score_filter_mode}. Expected 'dynamic', 'fixed', or 'none'."
                )

            if threshold <= 0.0:
                filtered_boxes.append(boxes)
                continue

            keep_mask = torch.as_tensor(
                score_data > threshold,
                device=boxes.device,
                dtype=torch.bool,
            )
            filtered_boxes.append(boxes[keep_mask])

        return filtered_boxes

    def _resolve_score_filter_policy(
        self,
        candidate_points_per_image,
        prior_mode,
        score_filter_mode,
        score_threshold,
    ):
        """Pick the score-filter policy, preserving legacy behavior by default."""
        if score_filter_mode is not None:
            return score_filter_mode, score_threshold

        resolved_prior_mode = prior_mode or getattr(self.cellfinder.args, "prior_mode", None)
        if candidate_points_per_image is not None and resolved_prior_mode == "strict":
            resolved_threshold = 0.3 if score_threshold is None else score_threshold
            return "fixed", resolved_threshold

        return "dynamic", score_threshold

    @torch.no_grad()
    def generate_bounding_boxes(
        self,
        images,
        device=None,
        candidate_points_per_image=None,
        candidate_valid_masks=None,
        prior_mode=None,
        score_filter_mode=None,
        score_threshold=None,
        query_output_mode="filtered",
        apply_candidate_mask=True,
    ):
        """
        Generates bounding boxes for the given images with configurable score filtering.

        Default policy:
        - legacy / no-prior path -> dynamic score filtering
        - strict prior path -> fixed threshold 0.3
        """
        if query_output_mode not in {"filtered", "candidate_aligned"}:
            raise ValueError(
                f"Unsupported query_output_mode={query_output_mode}. "
                "Expected 'filtered' or 'candidate_aligned'."
            )

        if query_output_mode == "candidate_aligned":
            if candidate_points_per_image is None:
                raise ValueError(
                    "candidate_aligned output requires candidate_points_per_image."
                )
            query_results = self.generate_query_outputs(
                images,
                device=device,
                candidate_points_per_image=candidate_points_per_image,
                candidate_valid_masks=candidate_valid_masks,
                prior_mode=prior_mode,
                apply_candidate_mask=apply_candidate_mask,
            )
            return self._candidate_aligned_boxes_from_query_results(query_results)

        transformed_imgs_anchor = self.sam_bbox_preprocessing(images, percentile=False)
        candidate_points, candidate_valid_mask = self._prepare_candidate_prior_batch(
            candidate_points_per_image=candidate_points_per_image,
            candidate_valid_masks=candidate_valid_masks,
            batch_size=transformed_imgs_anchor.shape[0],
            max_queries=self.cellfinder.args.num_query_position,
            device=transformed_imgs_anchor.device,
        )
        results = self.cellfinder.forward_inference(
            transformed_imgs_anchor,
            candidate_points=candidate_points,
            candidate_valid_mask=candidate_valid_mask,
            prior_mode=prior_mode,
            apply_candidate_mask=apply_candidate_mask,
        )

        boxes_per_heatmap = [x["boxes"] for x in results]
        pred_scores = [x["scores"] for x in results]
        resolved_mode, resolved_threshold = self._resolve_score_filter_policy(
            candidate_points_per_image=candidate_points_per_image,
            prior_mode=prior_mode,
            score_filter_mode=score_filter_mode,
            score_threshold=score_threshold,
        )
        return self._filter_boxes_by_scores(
            boxes_per_heatmap,
            pred_scores,
            score_filter_mode=resolved_mode,
            score_threshold=resolved_threshold,
        )

    @torch.no_grad()
    def generate_embeddings(
            self, images, existing_embeddings=None, device=None
    ):
        """
        Generates embeddings for the given images or uses existing embeddings if provided.
        """
        if existing_embeddings is None:
            transformed_imgs_anchor, paddings = self.prep_2(images, percentile=True)
            x = self.forward(transformed_imgs_anchor, return_preprocessed=False)
            return x, paddings
        else:
            # Use existing embeddings and compute paddings
            paddings = []
            for img in images:
                h, w = img.shape[-2:]
                paddings.append((1024 - h, 1024 - w))
            return existing_embeddings, paddings

    def predict(self, images, coords_per_heatmap=None, boxes_per_heatmap=None):
        device = next(self.parameters()).device

        assert self.mask_threshold > 0

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        x, paddings = self.generate_embeddings(images, device=device)

        if boxes_per_heatmap is None:
            boxes_per_heatmap = self.generate_bounding_boxes(images, device=device)
        else:
            # scale the boxes to 1024x1024, with paddings from above
            scaled_boxes_per_heatmap = []
            for idx in range(len(images)):
                boxes = boxes_per_heatmap[idx] if idx < len(boxes_per_heatmap) else boxes_per_heatmap[0]
                _boxes = []
                for box in boxes:
                    _box = [b.cpu().numpy() if hasattr(b, 'cpu') else b for b in box]
                    im_w = images[0].shape[2]
                    im_h = images[0].shape[1]
                    scale_x = 1024 / im_w
                    scale_y = 1024 / im_h
                    _box = [
                        _box[0] * scale_x,
                        _box[1] * scale_y,
                        _box[2] * scale_x,
                        _box[3] * scale_y,
                    ]
                    _boxes.append(_box)
                scaled_boxes_per_heatmap.append(torch.tensor(_boxes).to(device))
            boxes_per_heatmap = scaled_boxes_per_heatmap

        for idx in range(len(x)):
            boxes = boxes_per_heatmap[idx] if idx < len(boxes_per_heatmap) else boxes_per_heatmap[0]
            rng = len(boxes)
            low_masks = []
            low_masks_thresholded = []
            scores = []
            final_boxes = []
            mdl = self.model_cp if self.adv_mode else self.model

            for coord_idx in range(rng):
                bbox = boxes[coord_idx]
                input_box = torch.as_tensor(bbox).unsqueeze(0).unsqueeze(0)

                sparse_embeddings, dense_embeddings = mdl.prompt_encoder(
                    points=None,
                    boxes=input_box.to(device),
                    masks=None,
                )

                low_res_masks, iou_predictions = mdl.mask_decoder(
                    image_embeddings=x[idx].unsqueeze(0).to(device),
                    image_pe=mdl.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                low_res_masks = low_res_masks.detach().cpu()

                # threshold based on iou predictions
                if iou_predictions[0][0] < self.iou_threshold:
                    warnings.warn("Low IOU threshold, ignoring mask.")
                    continue

                low_res_masks = self.model.postprocess_masks(
                    low_res_masks,
                    input_size=torch.tensor([1024 - paddings[idx][0], 1024 - paddings[idx][1]]).to(device),
                    original_size=images[idx].shape[-2:]
                )
                low_res_masks_thresholded = nn.Sigmoid()(low_res_masks[0, 0]) > self.mask_threshold
                low_res_masks_thresholded = low_res_masks_thresholded.numpy().astype(np.uint8)

                res = low_res_masks[0, 0].detach().cpu().numpy()

                low_masks.append(res)
                low_res_masks_thresholded = low_res_masks_thresholded[:images[idx].shape[1], :images[idx].shape[2]]
                low_masks_thresholded.append(low_res_masks_thresholded)
                scores.append(float(iou_predictions[0][0].detach().cpu().numpy()))

                # Scale bbox back to original image size
                _bbox = [b.cpu().numpy() if hasattr(b, 'cpu') else b for b in bbox]
                im_w = images[0].shape[2]
                im_h = images[0].shape[1]
                scale_x = im_w / 1024
                scale_y = im_h / 1024
                _bbox = [
                    _bbox[0] * scale_x,
                    _bbox[1] * scale_y,
                    _bbox[2] * scale_x,
                    _bbox[3] * scale_y,
                ]
                final_boxes.append(_bbox)

            if low_masks:
                thresholded_masks = np.stack(low_masks_thresholded)
                final_boxes = np.stack(final_boxes)

                # Create instance segmentation mask
                thresholded_masks_summed = (
                        thresholded_masks * np.arange(1, thresholded_masks.shape[0] + 1)[:, None, None]
                )
                thresholded_masks_summed = np.max(thresholded_masks_summed, axis=0)

                return thresholded_masks_summed, thresholded_masks, x, final_boxes
            else:
                return None, None, None, None

    def load_state_dict(self, state_dict, strict=True):

        if isinstance(state_dict, dict) and 'state_dict' in state_dict and not any(
                k.startswith('model.') for k in state_dict.keys()):
            state_dict = state_dict['state_dict']
        has_model_cp = any(k.startswith('model_cp.') for k in state_dict.keys())
        result = super().load_state_dict(state_dict, strict=strict)
        if not has_model_cp:
            self.adv_mode = False
            self.model_cp.load_state_dict(self.model.state_dict(), strict=False)
        return result
