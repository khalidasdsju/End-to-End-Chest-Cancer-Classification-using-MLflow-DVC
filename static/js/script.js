document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const predictButton = document.getElementById('predict-button');
    const resultsSection = document.getElementById('results-section');
    const previewImage = document.getElementById('preview-image');
    const predictionResult = document.getElementById('prediction-result');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const loader = document.getElementById('loader');
    const newAnalysisButton = document.getElementById('new-analysis');

    // Variables
    let selectedFile = null;

    // Event Listeners
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('active');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('active');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('active');

        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });

    predictButton.addEventListener('click', handlePrediction);

    newAnalysisButton.addEventListener('click', resetAnalysis);

    // Functions
    function handleFileSelect(file) {
        // Check if file is an image
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        predictButton.disabled = false;

        // Preview the image
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    async function handlePrediction() {
        if (!selectedFile) return;

        // Show loader
        loader.style.display = 'block';
        predictionResult.innerHTML = '';
        resultsSection.style.display = 'block';

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                showError(data.error);
                return;
            }

            // Display results
            displayResults(data);

        } catch (error) {
            showError('An error occurred during prediction');
            console.error(error);
        } finally {
            loader.style.display = 'none';
        }
    }

    function displayResults(data) {
        // Set prediction text with appropriate styling
        let resultClass = data.prediction.toLowerCase() === 'malignant' ? 'danger' : 'success';

        // Check if in demo mode
        let demoModeText = '';
        if (data.demo_mode) {
            demoModeText = '<div class="demo-mode-badge"><i class="fas fa-flask"></i> Demo Mode</div>';
        }

        predictionResult.innerHTML = `
            <span class="${resultClass}">
                ${data.prediction}
                ${data.prediction.toLowerCase() === 'malignant' ?
                    '<i class="fas fa-exclamation-circle"></i>' :
                    '<i class="fas fa-check-circle"></i>'}
            </span>
            ${demoModeText}
        `;

        // Update confidence bar
        confidenceBar.style.width = `${data.confidence}%`;
        confidenceBar.style.backgroundColor = resultClass === 'danger' ? '#e74c3c' : '#2ecc71';
        confidenceText.textContent = `${data.confidence.toFixed(2)}%`;

        // Set image if provided
        if (data.image) {
            previewImage.src = `data:image/jpeg;base64,${data.image}`;
        }
    }

    function showError(message) {
        predictionResult.innerHTML = `<span class="error"><i class="fas fa-exclamation-triangle"></i> ${message}</span>`;
        confidenceBar.style.width = '0%';
        confidenceText.textContent = '0%';
    }

    function resetAnalysis() {
        // Reset UI
        resultsSection.style.display = 'none';
        fileName.textContent = 'None';
        predictButton.disabled = true;
        selectedFile = null;
        fileInput.value = '';

        // Scroll back to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // CSS classes for styling are defined in style.css
});
