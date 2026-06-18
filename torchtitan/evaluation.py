import gc
import torch
from pathlib import Path

from torchtitan.utils import dist_sum
from torchtitan.visualization import visualize_slices

def compute_pixel_accuracy(logits, targets):
    """
    Computes the percentage of correctly classified pixels.
    Args:
        logits: Model output tensor (B, C, H, W) or (B, C, D, H, W)
        targets: Ground truth tensor (B, H, W) or (B, D, H, W)
    """
    # Convert logits to class predictions
    preds = torch.argmax(logits, dim=1)
    
    # Count correct pixels
    correct = (preds == targets).sum().item()
    total = targets.numel()
    
    return correct / total

def compute_confusion_matrix(preds, targets, num_classes, ignore_index=None):
    """
    Computes the confusion matrix per volume for mIoU calculation.
    Args:
        preds: (D, H, W) - class predictions
        targets: (D, H, W) - ground truth labels
        num_classes: int
        ignore_index: int (optional) - Label to ignore
    Returns:
        conf_matrix: (num_classes, num_classes) tensor on the same device
    """    
    # Flatten inputs to 1D lists of pixels
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    # Filter out 'ignore_index' if necessary
    if ignore_index is not None:
        mask = (targets_flat != ignore_index)
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]
    
    # Compute confusion matrix using bincount
    unique_mapping = targets_flat * num_classes + preds_flat
    
    # Count occurrences
    hist = torch.bincount(unique_mapping, minlength=num_classes**2)
    
    # Reshape to (num_classes, num_classes)
    conf_matrix = hist.reshape(num_classes, num_classes)
    
    return conf_matrix.float()

def evaluate_2d(model, val_loader, job_config, loss_fn, resample_fn, dp_mesh):
    # Initialize the counters
    total_val_loss = 0.0
    num_val_samples = 0
    num_classes = job_config.model.num_classes
    conf_matrix_all = torch.zeros((num_classes, num_classes), device='cuda')

    # --- Streaming Buffers ---
    current_sample_id = None
    current_pred_vol = None
    current_gt_vol = None

    def process_completed_volume():
        """Helper function to compute metrics on the GPU and aggressively destroy the volume."""
        nonlocal current_pred_vol, current_gt_vol, conf_matrix_all
        
        if current_pred_vol is not None and current_gt_vol is not None:
            # 1. Average the predictions & take argmax over classes (Entirely on CUDA)
            pred_vol_labels = torch.argmax(current_pred_vol, dim=0)  # (D, H, W)
            
            # 2. Compute confusion matrix natively on the GPU
            batch_conf_matrix = compute_confusion_matrix(pred_vol_labels, current_gt_vol, num_classes)
            
            # 3. Accumulate (no .cuda() cast needed anymore)
            conf_matrix_all += batch_conf_matrix

            # 4. Explicitly DESTROY the massive float32 tensors to free VRAM immediately
            del current_pred_vol
            del current_gt_vol
            del pred_vol_labels

    # -------------------------
    for val_inputs, val_targets, val_metas in val_loader:
        val_inputs = val_inputs.cuda(non_blocking=True)
        val_targets = val_targets.cuda(non_blocking=True)
        
        val_preds = model(val_inputs)
        val_preds = resample_fn(val_preds, val_targets, job_config.model.crop_size)

        # 1. Val loss
        val_loss = loss_fn(val_preds, val_targets)
        total_val_loss += val_loss.item()
        num_val_samples += 1

        # 2. Building crop-wise (volumetric) predictions sequentially
        val_probs = torch.softmax(val_preds, dim=1)
        val_batch_size = val_inputs.size(0)
        
        for b in range(val_batch_size):
            sample_id = val_metas["sample_id"][b]
            axis_str = val_metas["axis"][b]
            slice_idx = int(val_metas["slice_idx"][b])
            vol_shape = tuple(val_metas["vol_shape"][b].tolist())
            
            # --- Boundary Detection ---
            if current_sample_id != sample_id:
                if current_sample_id is not None:
                    process_completed_volume()
                    # print(f"sample_id: {current_sample_id}")
                
                # Initialize fresh buffers for the new sample directly on CUDA
                current_sample_id = sample_id
                current_pred_vol = torch.zeros((num_classes,) + vol_shape, device='cuda')
                current_gt_vol = torch.zeros(vol_shape, dtype=torch.long, device='cuda')

            # --- Accumulate Predictions (Keeping data on GPU) ---
            # We still detach to prevent holding onto the autograd graph, but keep it on CUDA
            current_slice_probs = val_probs[b].detach()
            
            if axis_str == 'z':
                current_pred_vol[:, slice_idx, :, :] += current_slice_probs
                current_gt_vol[slice_idx, :, :] = val_targets[b].detach()
            elif axis_str == 'y':
                current_pred_vol[:, :, slice_idx, :] += current_slice_probs
            elif axis_str == 'x':
                current_pred_vol[:, :, :, slice_idx] += current_slice_probs

    # --- Process the final volume ---
    if current_sample_id is not None:
        process_completed_volume()
        # print(f"sample_id: {current_sample_id}")

    # Nullify references and force garbage collection before returning to training loop
    current_pred_vol = None
    current_gt_vol = None
    
    # We clear both CPU RAM (just in case) and the CUDA memory cache to ensure 
    # the GPU is perfectly clean before training resumes.
    gc.collect()
    torch.cuda.empty_cache()

    # Reduce val loss
    local_stats = torch.tensor([total_val_loss, num_val_samples], device='cuda', dtype=torch.float32)
    local_stats = dist_sum(local_stats, dp_mesh)

    global_total_loss = local_stats[0].item()
    global_total_samples = local_stats[1].item()
    avg_val_loss = global_total_loss / global_total_samples

    # Reduce confusion matrix across ranks
    conf_matrix_all = dist_sum(conf_matrix_all, dp_mesh)

    # Mean IoU calculation
    true_positive = torch.diag(conf_matrix_all)
    rows_sum = conf_matrix_all.sum(dim=1) 
    cols_sum = conf_matrix_all.sum(dim=0) 
    union = rows_sum + cols_sum - true_positive
    
    iou_per_class = true_positive / (union + 1e-6)
    avg_miou = iou_per_class[rows_sum > 0].mean().item()

    return avg_val_loss, avg_miou

# def evaluate_2d(model, val_loader, job_config, loss_fn, resample_fn, dp_mesh):
#     # Initialize the counters
#     total_val_loss = 0
#     num_val_samples = 0
#     conf_matrix_all = torch.zeros((job_config.model.num_classes, job_config.model.num_classes), device='cuda')

#     visuals_path = Path(job_config.job.dump_folder) / "visuals"
#     visuals_path.mkdir(parents=True, exist_ok=True)

#     predictions = {}
#     ground_truths = {}
#     raw_inputs = {}

#     for val_inputs, val_targets, val_metas in val_loader:
#         val_inputs = val_inputs.cuda()
#         val_targets = val_targets.cuda()
        
#         val_preds = model(val_inputs)
#         val_preds = resample_fn(val_preds, val_targets, job_config.model.crop_size)

#         # 1. Val loss
#         val_loss = loss_fn(val_preds, val_targets)
#         total_val_loss += val_loss.item()
#         num_val_samples += 1

#         # 2. Building crop-wise (volumetric) predictions
#         val_probs = torch.softmax(val_preds, dim=1)
#         val_batch_size = val_inputs.size(0)
        
#         for b in range(val_batch_size):
#             sample_id = val_metas["sample_id"][b]
#             axis_str = val_metas["axis"][b]
#             axis = {'z': 0, 'y': 1, 'x': 2}[axis_str]
#             slice_idx = int(val_metas["slice_idx"][b])
#             vol_shape = tuple(val_metas["vol_shape"][b].tolist())
            
#             # --- Initialize Buffers on CPU (Not CUDA) ---
#             if sample_id not in predictions:
#                 predictions[sample_id] = torch.zeros((job_config.model.num_classes,) + vol_shape, device='cpu')
#                 ground_truths[sample_id] = torch.zeros(vol_shape, dtype=torch.long, device='cpu')
#                 raw_inputs[sample_id] = torch.zeros(vol_shape, dtype=torch.float32, device='cpu')

#             # --- Accumulate Predictions ---
#             current_slice_probs = val_probs[b].detach().cpu()
            
#             if axis == 0:
#                 predictions[sample_id][:, slice_idx, :, :] += current_slice_probs
#             elif axis == 1:
#                 predictions[sample_id][:, :, slice_idx, :] += current_slice_probs
#             elif axis == 2:
#                 predictions[sample_id][:, :, :, slice_idx] += current_slice_probs
            
#             # --- Accumulate Labels ---
#             if axis == 0:
#                 ground_truths[sample_id][slice_idx, :, :] = val_targets[b]
#                 raw_inputs[sample_id][slice_idx, :, :] = val_inputs[b, 0, :, :].detach().cpu()
    
#     print(f"All dictionaries completed for evaluation!")
#     # crop-wise (voumetric predictions)
#     for sample_id, pred_vol in predictions.items():
#         # Average the predictions & take argmax over classes
#         pred_vol = torch.argmax(pred_vol, dim=0)  # (D, H, W)
        
#         # Retrieve the reconstructed 3D label
#         gt_vol = ground_truths[sample_id] # (D, H, W)

#         # Retrieve the raw inputs
#         raw_vol = raw_inputs[sample_id] # (D, H, W)

#         # mIoU
#         batch_conf_matrix = compute_confusion_matrix(pred_vol, gt_vol, job_config.model.num_classes)
#         conf_matrix_all += batch_conf_matrix.cuda()

#         safe_sample_id = str(sample_id).replace("/", "_").replace("\\", "_")

#         # visualize_slices(
#         #     raw_vol,
#         #     pred_vol,
#         #     gt_vol,
#         #     job_config.model.num_classes,
#         #     visuals_path / f"{job_config.model.backbone}_val_sample_{safe_sample_id}.gif"
#         # )

#     # Reduce val loss
#     local_stats = torch.tensor([total_val_loss, num_val_samples], device='cuda', dtype=torch.float32)
#     local_stats = dist_sum(local_stats, dp_mesh)

#     global_total_loss = local_stats[0].item()
#     global_total_samples = local_stats[1].item()
#     avg_val_loss = global_total_loss / global_total_samples

#     # Reduce confusion matrix across ranks
#     conf_matrix_all = dist_sum(conf_matrix_all, dp_mesh)

#     # Mean IoU calculation
#     true_positive = torch.diag(conf_matrix_all)
#     rows_sum = conf_matrix_all.sum(dim=1) 
#     cols_sum = conf_matrix_all.sum(dim=0) 
#     union = rows_sum + cols_sum - true_positive
    
#     iou_per_class = true_positive / (union + 1e-6)
#     avg_miou = iou_per_class[rows_sum > 0].mean().item()

#     return avg_val_loss, avg_miou

def evaluate_3d(model, val_loader, job_config, loss_fn, resample_fn, dp_mesh):
    # Initialize the counters
    total_val_loss = 0
    num_val_samples = 0
    conf_matrix_all = torch.zeros((job_config.model.num_classes, job_config.model.num_classes), device='cuda')  # initialize a confusion matrix on GPU

    # create the gif dump directory if it doesn't exist
    visuals_path = Path(job_config.job.dump_folder) / "visuals"
    visuals_path.mkdir(parents=True, exist_ok=True)

    rank = torch.distributed.get_rank()

    for val_inputs, val_targets in val_loader:
        val_inputs = val_inputs.cuda()
        val_targets = val_targets.cuda()
        val_preds = model(val_inputs)
        val_preds = resample_fn(val_preds, val_targets, job_config.model.crop_size)
        # logger.info(f"val inputs/targets/preds shape: {images.shape}/{labels.shape}/{outputs.shape}")

        # 1. Val loss
        val_loss = loss_fn(val_preds, val_targets)
        total_val_loss += val_loss.item()
        num_val_samples += 1

        # iterate over the batch to obtain crop-wise (voumetric) predictions
        for b in range(val_preds.shape[0]):
            # Average the predictions & take argmax over classes
            final_seg = torch.argmax(val_preds[b], dim=0)  # (D, H, W)
            
            # Retrieve the reconstructed 3D label
            gt_vol = val_targets[b] # (D, H, W)

            # Retrieve the raw inputs
            raw_vol = val_inputs[b, 0, :, :, :].detach().cpu()  # (D, H, W) this need not be on GPU

            # print(f"val_preds/val_preds/val_targets/raw_vol/final_seg/gt_vol: {val_inputs.shape}/{val_preds.shape}/{val_targets.shape}/{raw_vol.shape}/{final_seg.shape}/{gt_vol.shape}")

            # mIoU
            batch_conf_matrix = compute_confusion_matrix(final_seg, gt_vol, job_config.model.num_classes)
            conf_matrix_all += batch_conf_matrix

            # sample id to uniquely id crops
            sample_id = f"rank{rank}_batch{num_val_samples}_sample{b}"

            # Visualize results
            visualize_slices(
                raw_vol,
                final_seg,
                gt_vol,
                job_config.model.num_classes,
                visuals_path / f"{job_config.model.backbone}_val_sample_{sample_id}.gif"
            )

    # Reduce val loss
    local_stats = torch.tensor([total_val_loss, num_val_samples], device='cuda', dtype=torch.float32)
    local_stats = dist_sum(local_stats, dp_mesh)

    global_total_loss = local_stats[0].item()
    global_total_samples = local_stats[1].item()
    avg_val_loss = global_total_loss / global_total_samples

    # Reduce confusion matrix across ranks
    conf_matrix_all = dist_sum(conf_matrix_all, dp_mesh)

    # Mean IoU calculation
    true_positive = torch.diag(conf_matrix_all)
    rows_sum = conf_matrix_all.sum(dim=1) # Ground truth pixels per class
    cols_sum = conf_matrix_all.sum(dim=0) # Predicted pixels per class
    union = rows_sum + cols_sum - true_positive
    
    # Calculate IoU per class
    iou_per_class = true_positive / (union + 1e-6)
    
    # Mean IoU (ignoring classes that don't exist in targets)
    avg_miou = iou_per_class[rows_sum > 0].mean().item()

    return avg_val_loss, avg_miou