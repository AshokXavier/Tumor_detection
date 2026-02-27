document.addEventListener('DOMContentLoaded', () => {
    setupUploadForm('classifyForm', '/api/predict_2d', handleClassifyResult);
    setupUploadForm('segmentForm', '/api/predict_3d', handleSegmentResult);

    // Dynamic file preview for 2D
    const classifyInput = document.querySelector('#classifyForm #fileInput');
    if (classifyInput) {
        classifyInput.addEventListener('change', function (e) {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    document.getElementById('imagePreview').src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                    document.getElementById('analyzeBtn').disabled = false;
                    document.getElementById('fileName').textContent = classifyInput.files[0].name;
                }
                reader.readAsDataURL(this.files[0]);
            }
        });
    }

    // File name preview for 3D
    const segmentInput = document.querySelector('#segmentForm #fileInput');
    if (segmentInput) {
        segmentInput.addEventListener('change', function (e) {
            if (this.files && this.files[0]) {
                document.getElementById('fileInfo').style.display = 'block';
                document.getElementById('fileName').textContent = this.files[0].name;
                document.getElementById('analyzeBtn').disabled = false;
            }
        });
    }
});

function setupUploadForm(formId, endpoint, callback) {
    const form = document.getElementById(formId);
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);

        const loader = document.getElementById('loader');
        const resultContainer = document.getElementById('resultContainer');
        const btn = form.querySelector('button[type="submit"]');

        // UI State: Loading
        loader.style.display = 'block';
        if (resultContainer) resultContainer.style.display = 'none';
        btn.disabled = true;

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            callback(data);
            if (resultContainer) {
                resultContainer.classList.add('animate-fade-in');
                resultContainer.style.display = 'block';
            }

        } catch (error) {
            alert('Error: ' + error.message);
            console.error(error);
        } finally {
            loader.style.display = 'none';
            btn.disabled = false;
        }
    });
}

function handleClassifyResult(data) {
    const predictionEl = document.getElementById('predictionResult');
    const confidenceEl = document.getElementById('confidenceResult');

    // Multi-class display
    const label = data.label;
    const confidence = (data.confidence * 100).toFixed(2) + '%';

    predictionEl.textContent = label;

    // Color logic: "No Tumor" is safe (Green), others are warnings (Red)
    const isSafe = label.toLowerCase().includes('no tumor');
    predictionEl.className = isSafe ? 'badge-success' : 'badge-warning';
    
    confidenceEl.textContent = confidence;

    const confidenceBar = document.getElementById('confidenceBar');
    if (confidenceBar) {
        confidenceBar.style.width = confidence;
    }

    // Enable Report
    enableReportDownload('classifyForm', '/api/report_2d');
}

function handleSegmentResult(data) {
    if (data.metrics) {
        document.getElementById('totalVolume').textContent = data.metrics.total_volume_cm3;
        document.getElementById('necroticVolume').textContent = data.metrics.necrotic_cm3;
        document.getElementById('edemaVolume').textContent = data.metrics.edema_cm3;
        document.getElementById('enhancingVolume').textContent = data.metrics.enhancing_cm3;
    }

    if (data.image) {
        const img = document.getElementById('sliceImage');
        const placeholder = document.getElementById('slicePlaceholder');

        img.src = data.image;
        img.style.display = 'block';
        placeholder.style.display = 'none';
    }

    // Slice index display removed as per user request
    if (data.slice_index !== undefined) {
        const el = document.getElementById('sliceIndexDisplay');
        if (el) el.style.display = 'none'; // Hide if still exists
    }

    // Enable Report
    enableReportDownload('segmentForm', '/api/report_3d');

    // Trigger Mesh Load (if window.loadMesh is defined)
    const fileInput = document.querySelector('#segmentForm input[type="file"]');
    if (window.loadMesh && fileInput && fileInput.files[0]) {
        window.loadMesh(fileInput.files[0]);
    }
}

function enableReportDownload(formId, endpoint) {
    const btn = document.getElementById('downloadReportBtn');
    const form = document.getElementById(formId);
    if (btn && form) {
        btn.style.display = 'inline-block'; // Show button
        btn.onclick = async () => {
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';
            btn.disabled = true;

            const fileInput = form.querySelector('input[type="file"]');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const res = await fetch(endpoint, { method: 'POST', body: formData });
                if (res.ok) {
                    const blob = await res.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `Medical_Report_${fileInput.files[0].name.split('.')[0]}.pdf`;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    btn.innerHTML = '<i class="fa-solid fa-check"></i> Downloaded';
                } else {
                    throw new Error('Report generation failed');
                }
            } catch (e) {
                alert('Error: ' + e.message);
                btn.innerHTML = '<i class="fa-solid fa-file-pdf"></i> Retry Download';
                btn.disabled = false;
            }
        };
    }
}
