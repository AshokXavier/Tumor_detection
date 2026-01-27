from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import tensorflow as tf
import os
import io
import base64
from werkzeug.utils import secure_filename
import traceback
from utils import load_models, preprocess_image_2d, preprocess_volume_3d, calculate_tumor_metrics, generate_slice_visualization, predict_2d_model, predict_3d_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models on startup
print("Loading models...")
models = load_models()
print("Models loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/segment')
def segment():
    return render_template('segment.html')

@app.route('/api/predict_2d', methods=['POST'])
def predict_2d():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process and predict (predict function handles preprocessing)
            prediction = predict_2d_model(models['2d'], filepath)
            print(f"DEBUG: 2D Prediction Output: {prediction}, Shape: {prediction.shape}, Type: {type(prediction)}")
            
            # Multi-class handling
            # Model returns raw logits (no softmax activation in last layer)
            logits = prediction[0]
            probs = tf.nn.softmax(logits).numpy().tolist()
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])
            
            # Standard Brain Tumor Labels (Alphabetical/Common):
            # 0: Glioma, 1: Meningioma, 2: No Tumor, 3: Pituitary
            
            LABELS = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']
            label = LABELS[class_idx] if class_idx < len(LABELS) else f"Class {class_idx}"
            
            print(f"DEBUG: Parsed Class: {class_idx}, Label: {label}, Conf: {confidence}")
            
            result = {
                'prediction': probs,
                'class_index': class_idx,
                'confidence': confidence,
                'label': label
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/predict_3d', methods=['POST'])
def predict_3d():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 1. Predict
        segmentation = predict_3d_model(models['3d'], filepath)
        
        # 2. Extract voxel spacing from original file
        try:
            import nibabel as nib 
            img_obj = nib.load(filepath)
            dx, dy, dz = img_obj.header.get_zooms()
            voxel_spacing = (dx, dy, dz)
        except:
            voxel_spacing = (1.0, 1.0, 1.0)

        # 3. Metrics
        metrics = calculate_tumor_metrics(segmentation, voxel_spacing=voxel_spacing)
        
        # 4. Visualization
        processed_volume = preprocess_volume_3d(filepath)
        # Adapt for 3-channel models
        if models['3d'].input_shape[-1] == 3 and processed_volume.shape[-1] == 1:
            processed_volume = np.repeat(processed_volume, 3, axis=-1)
            
        slice_img_base64, max_slice = generate_slice_visualization(processed_volume, segmentation)
        
        return jsonify({
            'metrics': metrics,
            'image': slice_img_base64,
            'slice_index': int(max_slice),
            'message': 'Segmentation complete'
        })
    except Exception as e:
        print(f"3D Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/mesh', methods=['POST'])
def api_mesh():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File required'}), 400
            
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        segmentation = predict_3d_model(models['3d'], filepath)
        
        # Get spacing for scaling (convert numpy floats to native python floats)
        try:
            import nibabel as nib
            img_obj = nib.load(filepath)
            spacing = [float(s) for s in img_obj.header.get_zooms()]
        except:
            spacing = [1.0, 1.0, 1.0]
            
        # segmentation shape: (1, 128, 128, 128, 4)
        # Argmax to get 3D mask
        mask_3d = np.argmax(segmentation[0], axis=-1) # (128, 128, 128)
        
        from utils_mesh import generate_tumor_mesh_obj
        obj_content = generate_tumor_mesh_obj(mask_3d)
        
        if obj_content is None:
            return jsonify({'error': 'No tumor detected for 3D reconstruction'}), 404
            
        return jsonify({
            'obj': obj_content,
            'spacing': spacing
        })
        
    except Exception as e:
        print(f"Mesh API Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/api/compare', methods=['POST'])
def api_compare():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files are required'}), 400
        
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        from utils import generate_comparison_visualization
        
        # Helper to process a single file (reusing logic)
        def process_3d_file(file_obj):
            filename = secure_filename(file_obj.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_obj.save(filepath)
            
            model = models['3d']
            input_shape = model.input_shape
            target_shape = None
            if len(input_shape) == 5:
                target_shape = input_shape[1:4]
                
            processed_volume = preprocess_volume_3d(filepath, target_shape=target_shape)
             
            if len(input_shape) == 5:
                model_channels = input_shape[-1]
                data_channels = processed_volume.shape[-1]
                if model_channels == 3 and data_channels == 1:
                     processed_volume = np.repeat(processed_volume, 3, axis=-1)
            
            segmentation = model.predict(processed_volume)
            return processed_volume, segmentation
            
        vol1, seg1 = process_3d_file(file1)
        vol2, seg2 = process_3d_file(file2)
        
        # Generate comparison
        comparison_img, s1, s2 = generate_comparison_visualization(vol1, seg1, vol2, seg2)
        
        return jsonify({
            'image': comparison_img,
            'slice1': int(s1),
            'slice2': int(s2),
            'message': 'Comparison complete'
        })
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_gradcam', methods=['POST'])
def api_compare_gradcam():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files are required'}), 400
        
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        from utils import generate_gradcam_visualization
        
        # Helper
        def process_gradcam(file_obj):
            filename = secure_filename(file_obj.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_obj.save(filepath)
            
            # Predict just to get label (optional, but good for context)
            # but here we just need visualization
            img_base64 = generate_gradcam_visualization(filepath, models['2d'])
            return img_base64
            
        img1 = process_gradcam(file1)
        img2 = process_gradcam(file2)
        
        return jsonify({
            'image1': img1,
            'image2': img2,
            'message': 'Grad-CAM generated'
        })
        
    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/report_2d', methods=['POST'])
def api_report_2d():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File required'}), 400
            
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        prediction = predict_2d_model(models['2d'], filepath)
        
        # Consistent Softmax Handling (Matches api/predict_2d)
        logits = prediction[0]
        probs = tf.nn.softmax(logits).numpy()
        class_idx = int(np.argmax(probs))
        confidence_val = float(probs[class_idx])
        confidence_str = f"{confidence_val * 100:.2f}%"
        
        # Labels MUST match api/predict_2d order!
        # Standard: 0: Pituitary, 1: No Tumor, 2: Meningioma, 3: Glioma
        LABELS = ['Pituitary', 'No Tumor', 'Meningioma', 'Glioma']
        pred_label = LABELS[class_idx] if class_idx < len(LABELS) else f"Class {class_idx}"
        
        # Generate PDF
        from utils_report import generate_pdf_report_2d
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Report_2D_{filename}.pdf")
        generate_pdf_report_2d(pred_label, confidence_str, filepath, filename, pdf_path)
        
        return send_file(pdf_path, as_attachment=True)
        
    except Exception as e:
        print(f"Report 2D Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/report_3d', methods=['POST'])
def api_report_3d():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File required'}), 400
            
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Generating 3D Report for {filename}...")
        
        # Predict
        segmentation = predict_3d_model(models['3d'], filepath)
        
        # Get Spacing
        try:
             import nibabel as nib
             dx, dy, dz = nib.load(filepath).header.get_zooms()
             spacing = (dx, dy, dz)
        except:
             spacing = (1.0, 1.0, 1.0)
             
        metrics = calculate_tumor_metrics(segmentation, voxel_spacing=spacing)
        
        # Generate Visualization Image
        try:
            from utils import generate_slice_visualization
            processed_vol = preprocess_volume_3d(filepath)
            
            # Channel handling
            if models['3d'].input_shape[-1] == 3 and processed_vol.shape[-1] == 1:
                processed_vol = np.repeat(processed_vol, 3, axis=-1)
                
            b64_str, _ = generate_slice_visualization(processed_vol, segmentation)
            header, encoded = b64_str.split(',', 1)
            data = base64.b64decode(encoded)
            
            temp_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_report_img_3d.png')
            with open(temp_img_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(f"Visualization Generation Failed: {e}")
            traceback.print_exc()
            temp_img_path = None # Report will handle None image gracefully
            
        # Generate PDF
        try:
            from utils_report import generate_pdf_report_3d
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Report_3D_{filename}.pdf")
            generate_pdf_report_3d(metrics, temp_img_path, filename, pdf_path)
            
            print(f"PDF Generated at {pdf_path}")
            return send_file(pdf_path, as_attachment=True)
        except Exception as pdf_err:
            print(f"PDF Generation Failed: {pdf_err}")
            traceback.print_exc()
            return jsonify({'error': 'PDF Generation failed after analysis'}), 500
        
    except Exception as e:
        print(f"Report 3D Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def progress():
    return render_template('progress.html')

@app.route('/api/progress', methods=['POST'])
def api_progress():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No selected files'}), 400
            
        results = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 1. Predict
            segmentation = predict_3d_model(models['3d'], filepath)
            
            # 2. Get Spacing
            try:
                import nibabel as nib
                dx, dy, dz = nib.load(filepath).header.get_zooms()
                spacing = (dx, dy, dz)
            except:
                spacing = (1.0, 1.0, 1.0)
                
            # 3. Calculate Metrics
            metrics = calculate_tumor_metrics(segmentation, voxel_spacing=spacing)
            
            results.append({
                'filename': filename,
                'total_volume_mm3': metrics['total_volume_mm3'],
                'metrics': metrics
            })
            
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Progress API Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/olmo', methods=['POST'])
def api_chat_olmo():
    try:
        data = request.json
        message = data.get('message', '').lower().strip('?!. ')
        
        # 1. CONVERSATIONAL / GENERAL RESPONSES
        general_responses = {
            "hello": "Hello! I'm Olmo, your medical imaging assistant. How's your day going?",
            "hi": "Hi there! I'm Olmo. Ready to look at some scans?",
            "how are you": "I'm functioning at 100% capacity and ready to analyze some MRIs! How about you?",
            "who are you": "I'm Olmo, a specialized AI built to help you navigate TumorVision. I'm named after the majestic Olmo tree, symbolizing wisdom and protection.",
            "what can you do": "I can explain 3D tumor mapping, help you track tumor growth over time, or walk you through the 2D classification results. Just ask!",
            "thank": "You're very welcome! I'm here to make this complex data easier to understand.",
            "bye": "Goodbye! I'll be here whenever you need another analysis.",
            "weather": "I don't have a window, but it's always a clear day inside the server! Best to check your local forecast.",
            "joke": "Why did the MRI scanner get a promotion? Because it was outstanding in its field (and very attractive!).",
            "time": f"In the server world, it's always 'Analysis Time'! But your local clock should be accurate.",
            "help": "I can help with: \n1. 3D Color Explanations\n2. Progress Tracking help\n3. Understanding AI Confidence\n4. General small talk!"
        }

        # 2. SYSTEM / CLINICAL KNOWLEDGE
        system_knowledge = {
            "classification": "Classification uses our 2D ensemble model. It identifies if a tumor is a Glioma, Meningioma, or Pituitary tumor. We use Grad-CAM heatmaps to show you exactly where the AI is looking.",
            "segmentation": "Segmentation is the 3D part. We use a 3D U-Net to map every voxel. The result is a rotatable 3D model that clearly shows the tumor sub-regions.",
            "progress": "Longitudinal Progress Tracking compares 'Visit 1' to 'Visit 2' to calculate if a tumor has regressed (shrunk) or progressed (grown).",
            "3d color": "Each color has a medical meaning:\n- **Red**: Enhancing Tumor (Active zones)\n- **Yellow**: Edema (Brain swelling)\n- **Cyan**: Necrotic Core (Dead tissue center)",
            "yellow": "Yellow represents Edema. This is fluid accumulation around the tumor that causes pressure. It's often what causes headaches for patients.",
            "red": "Red is the data-enhancing region. This is where the tumor is most active and consuming blood supply.",
            "blue": "Cyan/Blue is the Necrotic center. Aggressive tumors often outgrow their blood supply, causing the center to die off.",
            "cyan": "Cyan/Blue is the Necrotic center. Aggressive tumors often outgrow their blood supply, causing the center to die off.",
            "mri": "MRI stands for Magnetic Resonance Imaging. It uses strong magnets and radio waves to create detailed images of the brain without using radiation.",
            "glioma": "A Glioma is a type of tumor that starts in the glial cells of the brain or spine. They are the most common type of primary brain tumor.",
            "meningioma": "A Meningioma is a tumor that arises from the meninges — the membranes that surround your brain and spinal cord. They are usually slow-growing.",
            "accuracy": "Our 2D model has a validation accuracy of 98.5%, while the 3D segmentation achieves a Dice Score of 0.89.",

        }

        # Logic: Check specific system keywords first, then general conversation
        response = ""
        
        # Check System Keywords (Prioritize)
        found_system = False
        for key in system_knowledge:
            if key in message:
                response = system_knowledge[key]
                found_system = True
                break
        
        # Check General Keywords
        if not found_system:
            for key in general_responses:
                if key in message:
                    response = general_responses[key]
                    break
        
        # Fallback
        if not response:
            if len(message) < 3:
                response = "I'm listening! You can ask me things like 'What do the colors mean?' or just say 'Tell me a joke'."
            else:
                response = "That's an interesting question! While I'm specialized in the TumorVision system and Brain MRI analysis, I'm always learning. Could you rephrase that, or ask me about our 3D segmentation tools?"

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=7860)
