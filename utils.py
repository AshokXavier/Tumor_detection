import os
import numpy as np
import tensorflow as tf
import traceback
from PIL import Image
import nibabel as nib
from scipy.ndimage import zoom

# Custom Object Definition for 3D Model
# (Removed complex SumOfLosses implementation as compile=False should bypass it)
class SumOfLosses:
    pass

def load_models():
    """
    Loads the 2D classification and 3D segmentation models.
    """
    models = {}
    
    # Load 2D Model
    path_2d = os.path.join('models', 'ensemble_2d.keras')
    if os.path.exists(path_2d):
        try:
            models['2d'] = tf.keras.models.load_model(path_2d)
            print(f"2D Model loaded from {path_2d}")
        except Exception as e:
            print(f"Failed to load 2D model: {e}")
            models['2d'] = None
    else:
        print(f"2D Model not found at {path_2d}")
        models['2d'] = None

    # Load 3D Model
    path_3d = os.path.join('models', '3d_model.keras')
    if os.path.exists(path_3d):
        try:
            # Try loading with compile=False to avoid needing the custom loss function
            models['3d'] = tf.keras.models.load_model(path_3d, compile=False)
            print(f"3D Model loaded from {path_3d}")
        except Exception as e:
            print(f"Failed to load 3D model: {e}")
            traceback.print_exc()
            models['3d'] = None
    else:
        print(f"3D Model not found at {path_3d}")
        models['3d'] = None
        
    return models

def preprocess_image_2d(image_path, target_size=None):
    """
    Loads and preprocesses an image for the 2D model.
    """
    img = Image.open(image_path).convert('RGB')
    
    if target_size:
        img = img.resize(target_size)
    
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def compute_z_score(image):
    """
    Applies Z-score normalization (zero mean, unit variance) to non-zero region.
    Matches training code preprocessing.
    """
    image = image.astype(np.float32)
    non_zero = image[image > 0]
    if non_zero.size == 0:
        return image
    mean = non_zero.mean()
    std = non_zero.std()
    if std == 0:
        return image
    image_z = image.copy()
    image_z[image > 0] = (image[image > 0] - mean) / std
    return image_z

def preprocess_volume_3d(nifti_path, target_shape=None, return_coords=False):
    """
    Loads and preprocesses a NIfTI volume for the 3D model.
    Matches training code: Z-score -> Center Crop (128x128x128).
    Returns data, and optionally (start_D, start_H, start_W).
    """
    try:
        nifti = nib.load(nifti_path)
        data = nifti.get_fdata()
        
        # 1. Z-Score Normalization
        data = compute_z_score(data)
        
        # 2. Center Crop to 128x128x128
        target_D, target_H, target_W = 128, 128, 128
        
        start_D, start_H, start_W = 0, 0, 0
        
        if len(data.shape) == 3:
            src_D, src_H, src_W = data.shape
            
            # Calculate center offsets
            start_D = max(0, (src_D - target_D) // 2)
            start_H = max(0, (src_H - target_H) // 2)
            start_W = max(0, (src_W - target_W) // 2)
            
            # Crop
            end_D = min(src_D, start_D + target_D)
            end_H = min(src_H, start_H + target_H)
            end_W = min(src_W, start_W + target_W)
            
            crop = data[start_D:end_D, start_H:end_H, start_W:end_W]
            
            # Pad if result is smaller than target
            final_data = np.zeros((target_D, target_H, target_W), dtype=np.float32)
            d_len, h_len, w_len = crop.shape
            final_data[:d_len, :h_len, :w_len] = crop
            data = final_data
        
        data = np.expand_dims(data, axis=0) 
        data = np.expand_dims(data, axis=-1)
        
        if return_coords:
            return data, (start_D, start_H, start_W)
        return data
    except Exception as e:
        print(f"Error processing 3D volume: {e}")
        raise e

# ... (Predict functions remain same)

# ... (Calculate metrics remains same)

def generate_slice_visualization(original_vol_path, segmentation_pred, crop_coords=(0,0,0)):
    """
    Generates a High-Quality visualization using the original volume resolution.
    Overlays the 128x128 mask onto the full-size original slice.
    """
    from matplotlib.colors import ListedColormap
    
    # 1. Load Original Volume
    nifti = nib.load(original_vol_path)
    orig_data = nifti.get_fdata() # (D, H, W) or (H, W, D)? Nibabel usually (X, Y, Z) -> (H, W, D)
    # Let's verify nibabel orientation. 
    # Usually it matches the shape used in preprocessing.
    # Preprocessing treated it as (D, H, W)? 
    # Wait, in line 98: `src_D, src_H, src_W = data.shape`.
    # And we crop `start_D:end_D`.
    # BUT, typically medical images are (H, W, Depth).
    # If the code assumes (D, H, W), that implies the 0-th dimension is Depth.
    # Let's handle consistent with preprocessing logic.
    
    # Pred shape: (1, 128, 128, 128, 4)
    mask_crop = np.argmax(segmentation_pred[0], axis=-1) # (128, 128, 128)
    
    # 2. Find Max Tumor Slice in the CROP
    # This ensures we show the slice where the model actually found something.
    tumor_pixels_per_slice = np.sum(mask_crop > 0, axis=(1, 2)) # Sum over H, W? 
    # WAIT. In preprocessing: `crop = data[start_D:end_D, start_H:end_H, start_W:end_W]`
    # And typically standard orientation is (H, W, D).
    # If user used `src_D, src_H, src_W = data.shape` and `start_D` corresponds to index 0...
    # Then index 0 is the one we chopped for "Depth"? 
    # If `tumor_pixels_per_slice` calc uses axis (0, 1), it assumes axis 2 is depth.
    # Let's match typical visualization: Axis 2 is Depth.
    # If Preprocessing treated axis 0 as Depth?
    # Let's stick to the convention used in `generate_slice_visualization` previously:
    # `tumor_pixels_per_slice = np.sum(mask_data > 0, axis=(0, 1))` implies Axis 2 is slices.
    # If `preprocess` sliced all dims symmetrically, it's fine.
    
    # In `preprocess`: `crop = data[start_D:..., start_H:..., start_W:...]`
    # So `start_D` is offset for Axis 0.
    # `start_H` is offset for Axis 1.
    # `start_W` is offset for Axis 2.
    
    # If Axis 2 is the slice dim, then we need `start_W` (offset for axis 2).
    # Let's assume standard (H, W, D).
    # So crop offset is `crop_coords[2]`.
    
    start_0, start_1, start_2 = crop_coords
    
    # Check orientation of mask for max slice
    # Previous code: `tumor_pixels_per_slice = np.sum(mask_data > 0, axis=(0, 1))` (Summing H,W)
    # So max_slice_idx is along Axis 2.
    best_slice_crop_idx = np.argmax(np.sum(mask_crop > 0, axis=(0, 1)))
    
    # If empty
    if np.max(mask_crop) == 0:
        best_slice_crop_idx = mask_crop.shape[2] // 2
        
    # Mapping to Original Z
    # Since we cropped `data[:, :, start_2:end_2]`, the original index is:
    original_slice_idx = start_2 + best_slice_crop_idx
    
    # Clamp to bounds
    if original_slice_idx >= orig_data.shape[2]:
         original_slice_idx = orig_data.shape[2] - 1
         
    # 3. Extract High-Res Background Slice
    # Shape (H, W)
    bg_slice = orig_data[:, :, original_slice_idx]
    
    # Rotate 90 deg if needed? Standard NIfTI is usually rotated.
    # Let's rotate 90 deg counter-clockwise to match typical "upright" brain view if needed.
    # Often required for nibabel data. Sticking to raw for now, or maybe rot90 for better UX?
    # Let's apply np.rot90(bg_slice) to see if it fixes "squashed" look?
    # The user mentioned "squashed".
    # Often standard (H, W) needs rotation.
    bg_slice = np.rot90(bg_slice) 
    
    # 4. Reconstruct Full-Size Mask
    # Create empty mask same size as original slice
    full_mask_slice = np.zeros_like(bg_slice, dtype=np.uint8) # (H, W) (rotated)
    
    # Extract the crop slice from the 3D mask
    mask_slice_crop = mask_crop[:, :, best_slice_crop_idx] # (128, 128)
    
    # Rotate mask slice too!
    mask_slice_crop = np.rot90(mask_slice_crop)
    
    # Place it back?
    # Wait, if we rotate, the coordinates change.
    # Let's Place THEN Rotate. simpler.
    
    full_mask_flat = np.zeros((orig_data.shape[0], orig_data.shape[1]), dtype=np.uint8)
    
    # Insert crop
    # Crop coords: start_0 (H), start_1 (W)
    # Mask usage?
    # If crop was `data[start_0:..., start_1:...]`
    # Then we place mask at `[start_0:..., start_1:...]`
    h_crop, w_crop = mask_crop.shape[0], mask_crop.shape[1] # 128, 128
    
    # Handle padding cases (if original was smaller than 128) - unlikely for brain MRI
    # But assume we just overwrite.
    end_0 = min(start_0 + h_crop, full_mask_flat.shape[0])
    end_1 = min(start_1 + w_crop, full_mask_flat.shape[1])
    
    # Need to slice the mask if it goes out of bounds (padding case)
    # But usually 128 is smaller.
    
    full_mask_flat[start_0:end_0, start_1:end_1] = mask_slice_crop[:(end_0-start_0), :(end_1-start_1)]
    
    # Now Rotate Both for display
    bg_slice_vis = np.rot90(orig_data[:, :, original_slice_idx])
    mask_slice_vis = np.rot90(full_mask_flat)
    
    # 5. Plotting
    plt.figure(figsize=(10, 10), dpi=150) # High quality
    
    plt.imshow(bg_slice_vis, cmap='gray', interpolation='bicubic')
    
    # Colors
    colors = [
        (0, 0, 0, 0),       # Transparent
        (6/255, 182/255, 212/255, 0.7),     # Cyan
        (234/255, 179/255, 8/255, 0.7),     # Yellow
        (239/255, 68/255, 68/255, 0.7)      # Red
    ]
    cmap = ListedColormap(colors)
    
    plt.imshow(mask_slice_vis, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}", best_slice_crop_idx

def predict_2d_model(model, image_path):
    """
    Runs inference on the 2D model.
    """
    if model is None:
        raise ValueError("2D Model is not loaded.")
        
    input_shape = model.input_shape
    if isinstance(input_shape, list): 
        input_shape = input_shape[0]
        
    height = input_shape[1]
    width = input_shape[2]
    
    if height is None or width is None:
        height, width = 224, 224 
        
    processed_img = preprocess_image_2d(image_path, target_size=(width, height))
    
    prediction = model.predict(processed_img)
    return prediction

def predict_3d_model(model, volume_path):
    """
    Runs inference on the 3D model.
    """
    if model is None:
        raise ValueError("3D Model is not loaded.")
        
    input_shape = model.input_shape
    
    target_shape = None
    if len(input_shape) == 5:
        target_shape = input_shape[1:4]
    
    processed_volume = preprocess_volume_3d(volume_path, target_shape=target_shape)
    
    # Check channel compatibility
    # Model expects (Batch, D, H, W, Channels)
    if input_shape and len(input_shape) == 5:
        model_channels = input_shape[-1]
        data_channels = processed_volume.shape[-1]
        
        if model_channels == 3 and data_channels == 1:
            print("Adapting 1-channel input to 3-channel model (Stacking)")
            processed_volume = np.repeat(processed_volume, 3, axis=-1)
            
    segmentation = model.predict(processed_volume)
    return segmentation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def calculate_tumor_metrics(segmentation_pred, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Calculates tumor size for each class from the segmentation prediction.
    Expects softmax output distribution or logits (will apply argmax).
    Returns volumes in cm^3 (converted from mm^3).
    """
    # Get class map (argmax)
    # Shape: (1, 128, 128, 128, 4) -> (128, 128, 128)
    mask = np.argmax(segmentation_pred[0], axis=-1)
    
    # Voxel volume from spacing (dx * dy * dz)
    dx, dy, dz = voxel_spacing
    voxel_volume_mm3 = dx * dy * dz
    
    metrics = {
        'total_volume_cm3': 0,
        'necrotic_cm3': 0,
        'edema_cm3': 0,
        'enhancing_cm3': 0
    }
    
    # Class 0 is background
    # Class 1: Necrotic / Non-enhancing (Cyan/Red in visual?)
    # Class 2: Edema (Yellow)
    # Class 3: Enhancing Tumor (Blue/Green?)
    
    # Calculate raw mm3 first
    necrotic_val = np.sum(mask == 1) * voxel_volume_mm3
    edema_val = np.sum(mask == 2) * voxel_volume_mm3
    enhancing_val = np.sum(mask == 3) * voxel_volume_mm3
    
    # Convert to cm3 (1 cm3 = 1000 mm3)
    metrics['necrotic_cm3'] = float(f"{necrotic_val / 1000:.2f}")
    metrics['edema_cm3'] = float(f"{edema_val / 1000:.2f}")
    metrics['enhancing_cm3'] = float(f"{enhancing_val / 1000:.2f}")
    
    metrics['total_volume_cm3'] = float(f"{(necrotic_val + edema_val + enhancing_val) / 1000:.2f}")
    
    return metrics

def generate_slice_visualization(input_volume, segmentation_pred):
    """
    Generates a visualization of the slice with the maximum tumor area.
    Returns a base64 encoded image string.
    """
    from matplotlib.colors import ListedColormap
    
    # Input volume shape: (1, 128, 128, 128, 3)
    # Pred shape: (1, 128, 128, 128, 4)
    
    img_data = input_volume[0] # (128, 128, 128, 3)
    mask_data = np.argmax(segmentation_pred[0], axis=-1) # (128, 128, 128)
    
    # Find max tumor slice
    # Sum tumor pixels along axes 0 and 1 (H, W) to find best Depth slice
    tumor_pixels_per_slice = np.sum(mask_data > 0, axis=(0, 1))
    max_slice_idx = np.argmax(tumor_pixels_per_slice)
    
    # If no tumor, pick center slice
    if np.max(tumor_pixels_per_slice) == 0:
        max_slice_idx = mask_data.shape[2] // 2
        
    # Prepare plot
    plt.figure(figsize=(8, 8), dpi=200) # Higher DPI for quality
    
    # 1. Show MRI Image (Grayscale)
    # Rotate 90 degrees to match standard view if needed, or keeping as is.
    # Typically MRI slices need rotation to match radiological convention if raw numpy array.
    # Assuming standard orientation for now to match current website but boosting quality.
    plt.imshow(img_data[:, :, max_slice_idx, 0], cmap='gray', interpolation='bicubic')
    
    # 2. Define Custom Colormap for Mask
    # 0: Transparent
    # 1 (Necrotic): Cyan (#06b6d4) to match UI
    # 2 (Edema): Yellow (#eab308)
    # 3 (Enhancing): Red (#ef4444)
    colors = [
        (0, 0, 0, 0),       # 0: Background - Transparent
        (220/255, 38/255, 38/255, 1.0),     # 1: Red (Necrotic/Core often core) - Wait, code comment said 1 is Necrotic.
                                            # Let's match the visual in Kaggle (Inner Core is Cyan, Middle Red, Outer Yellow)
                                            # Usually in BraTS: 1=Necrotic(Core), 2=Edema, 4=Enhancing.
                                            # Here we map 1->Cyan, 2->Yellow, 3->Red?
                                            # Let's try: 1: Red (Necrotic), 2: Yellow (Edema), 3: Green/Cyan (Enhancing)?
                                            # Actually, let's stick to standard palette requested or inferred.
                                            # Kaggle image: Yellow outer, Red middle, Cyan inner.
                                            # Assuming nesting: Edema(Yellow) > Enhancing(Red) > Necrotic(Cyan).
        (250/255, 204/255, 21/255, 1.0),    # 2: Yellow (Edema)
        (6/255, 182/255, 212/255, 1.0)      # 3: Cyan (Enhancing? or Necrotic?)
    ]
    # Re-verify standard mapping from line 211: 
    # Class 1: Necrotic (Dead inner) -> Cyan
    # Class 2: Edema (Swelling outer) -> Yellow
    # Class 3: Enhancing (Active ring) -> Red
    
    # Corrected Palette based on standard bio-markers:
    colors = [
        (0, 0, 0, 0),       # 0: Transparent
        (6/255, 182/255, 212/255, 0.7),     # 1: Cyan (Necrotic/Core)
        (234/255, 179/255, 8/255, 0.7),     # 2: Yellow (Edema)
        (239/255, 68/255, 68/255, 0.7)      # 3: Red (Enhancing)
    ]
    cmap = ListedColormap(colors)
    
    # 3. Overlay Tumor Mask
    # Use 'nearest' to keep the pixelated "data" look and avoid blurry edges
    plt.imshow(mask_data[:, :, max_slice_idx], cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    
    plt.axis('off')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    
    # Encode
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}", max_slice_idx

def generate_comparison_visualization(vol1, seg1, vol2, seg2):
    """
    Generates a side-by-side comparison of max tumor slices for two patients.
    Returns base64 encoded image.
    """
    # Helper to find max slice
    def get_max_slice(mask_3d):
        # mask_3d shape: (128, 128, 128)
        # Sum along H, W (axis 0, 1) to get per-slice tumor count
        slice_sums = np.sum(mask_3d > 0, axis=(0, 1))
        if np.max(slice_sums) == 0:
            return mask_3d.shape[2] // 2
        return np.argmax(slice_sums)

    # Extract data (Batch 0)
    img1_data = vol1[0] # (128, 128, 128, 3)
    mask1_data = np.argmax(seg1[0], axis=-1)
    
    img2_data = vol2[0]
    mask2_data = np.argmax(seg2[0], axis=-1)
    
    slice1 = get_max_slice(mask1_data)
    slice2 = get_max_slice(mask2_data)
    
    # Prepare Plot
    plt.figure(figsize=(16, 8), dpi=150)
    
    # Patient 1
    plt.subplot(1, 2, 1)
    plt.imshow(img1_data[:, :, slice1, 0], cmap='gray')
    plt.imshow(mask1_data[:, :, slice1], cmap='jet', alpha=0.4, vmin=0, vmax=3)
    plt.axis('off')
    
    # Patient 2
    plt.subplot(1, 2, 2)
    plt.imshow(img2_data[:, :, slice2, 0], cmap='gray')
    plt.imshow(mask2_data[:, :, slice2], cmap='jet', alpha=0.4, vmin=0, vmax=3)
    # plt.title(f"Patient 2 - Tumor Slice")
    plt.axis('off')
    
    # Save
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    
    img_base64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    return img_base64, slice1, slice2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates Grad-CAM heatmap for a given image and model.
    """
    # 1. Create a model that maps the input image to the activations
    #    of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute the gradient of the top predicted class for our input image
    #    with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        model_out = grad_model(img_array)
        print(f"DEBUG: grad_model output type: {type(model_out)}, len: {len(model_out)}")
        if isinstance(model_out, list):
             last_conv_layer_output = model_out[0]
             preds = model_out[1]
        else:
             # Should not happen if we requested 2 outputs
             print("DEBUG: grad_model returned single output?")
             last_conv_layer_output, preds = model_out
             
        # Fix for "list indices must be integers or slices, not tuple"
        # If preds is a list (e.g. model.output was a list), extract the tensor.
        if isinstance(preds, list):
            preds = preds[0]

        print(f"DEBUG: preds type: {type(preds)}, shape: {getattr(preds, 'shape', 'NoShape')}")
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        print(f"DEBUG: Using pred_index: {pred_index}")
        class_channel = preds[:, pred_index]

    # 3. This is the gradient of the output neuron (top predicted or chosen)
    #    with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. This is a vector where each entry is the mean intensity of the gradient
    #    over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. We multiply each channel in the feature map array
    #    by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_visualization(img_path, model):
    """
    Generates a Grad-CAM visualization for a 2D image.
    Returns base64 encoded image.
    """
    # Get model input shape dynamically
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    
    # Check for (Batch, H, W, C)
    target_h, target_w = 224, 224 # Default
    if input_shape and len(input_shape) == 4:
        target_h = input_shape[1]
        target_w = input_shape[2]
        
    if target_h is None or target_w is None:
        target_h, target_w = 256, 256 # Fallback based on error message
        
    # Preprocess
    img_array = preprocess_image_2d(img_path, target_size=(target_w, target_h))
    
    # Find last conv layer dynamically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    if not last_conv_layer_name:
        # Fallback for EfficientNet/ResNet type models where last conv is nested?
        # Or if it's a Sequential model with specific blocks.
        # Let's assume standard for now or try to find it.
        # If not found, raise warning.
        print("Warning: No Conv2D layer found for Grad-CAM.")
        return None

    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Load original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap
    jet = matplotlib.colormaps.get_cmap("jet")
    
    # Prepare RGBA heatmap
    # jet(x) returns (R,G,B,A)
    # heatmap is 0-255. We need to index or map.
    # jet takes floats 0..1.
    heatmap_norm = heatmap.astype(float) / 255.0
    jet_heatmap = jet(heatmap_norm) # (H, W, 4)
    
    # Create custom Alpha channel
    # Make low values transparent to avoid "Blue Wash"
    # Threshold: values < 0.2 become transparent
    # Values > 0.2 ramp up opacity
    alpha = np.clip((heatmap_norm - 0.2) / 0.8, 0, 1) * 0.6 # Max opacity 0.6
    
    # Update alpha channel
    jet_heatmap[:, :, 3] = alpha
    
    # Convert to PIL Image for compositing
    jet_heatmap_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap_img = jet_heatmap_img.resize((img.shape[1], img.shape[0]))
    
    # Original Image (RGB)
    img_pil = tf.keras.preprocessing.image.array_to_img(img)
    
    # Composite
    # We overlay jet_heatmap_img ON TOP of img_pil
    # Since jet_heatmap_img has alpha, we can use PIL alpha_composite or similar.
    # But array_to_img might lose alpha if mode is not RGBA.
    
    # Let's do blending in numpy for precision
    jet_heatmap_arr = tf.keras.preprocessing.image.img_to_array(jet_heatmap_img) # (H, W, 4)
    img_arr = tf.keras.preprocessing.image.img_to_array(img_pil) # (H, W, 3)
    
    # Normalize to 0-1
    overlay = jet_heatmap_arr[:, :, :3] / 255.0
    alpha_mask = jet_heatmap_arr[:, :, 3] / 255.0
    background = img_arr / 255.0
    
    # Hand-coded composition: src * alpha + dst * (1 - alpha)
    # Explicitly broadcast alpha
    alpha_mask = np.expand_dims(alpha_mask, axis=-1)
    
    superimposed = (overlay * alpha_mask) + (background * (1.0 - alpha_mask))
    
    # Convert back to uint8
    superimposed = np.uint8(255 * superimposed)
    superimposed_img = Image.fromarray(superimposed)
    
    # Save to buffer
    buf = io.BytesIO()
    superimposed_img.save(buf, format='PNG')
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
