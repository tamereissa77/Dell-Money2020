const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const extractBtn = document.getElementById('extract-btn');
const imagePreview = document.getElementById('image-preview');
const previewContainer = document.getElementById('preview-container');
const fileInfo = document.getElementById('file-info');
const statusBadge = document.getElementById('status-container');
const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsContent = document.getElementById('results-content');
const dataDisplay = document.getElementById('data-display');
const jsonOutput = document.getElementById('json-output');
const toggleJsonBtn = document.getElementById('toggle-json');

let selectedFile = null;

// Drag and drop handlers
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    selectedFile = file;
    fileInfo.textContent = file.name;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        dropZone.style.display = 'none';
        extractBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    
    // Reset results
    resultsContent.style.display = 'none';
    resultsPlaceholder.style.display = 'block';
    statusBadge.textContent = 'Ready';
    statusBadge.className = 'status-badge';
}

extractBtn.addEventListener('click', async () => {
    const typeSelect = document.getElementById('doc-type');
    const selectedType = typeSelect.value;
    const typeText = typeSelect.options[typeSelect.selectedIndex].text;

    // UI Updates
    extractBtn.disabled = true;
    extractBtn.textContent = 'Processing...';
    statusBadge.textContent = `Analyzing ${typeText}...`;
    statusBadge.classList.add('processing');
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('doc_type', selectedType);

    try {
        console.log(`Sending extraction request for: ${selectedType}`);
        const response = await fetch('/extract', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.status === 'success') {
            displayResults(result.data);
            statusBadge.textContent = 'Complete';
            statusBadge.className = 'status-badge';
        } else {
            throw new Error(result.message || 'Extraction failed');
        }
    } catch (err) {
        console.error(err);
        statusBadge.textContent = 'Error';
        statusBadge.classList.remove('processing');
        alert('Error during extraction: ' + err.message);
    } finally {
        extractBtn.disabled = false;
        extractBtn.textContent = 'Start Extraction';
    }
});

function displayResults(data) {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';
    dataDisplay.innerHTML = '';
    
    // Populate the clean grid
    for (const [key, value] of Object.entries(data)) {
        if (value === null || value === undefined) continue;
        
        const item = document.createElement('div');
        item.className = 'data-item';
        
        const label = document.createElement('div');
        label.className = 'data-label';
        label.textContent = key.replace(/_/g, ' ');
        
        const valDiv = document.createElement('div');
        valDiv.className = 'data-value';
        valDiv.textContent = value;
        
        item.appendChild(label);
        item.appendChild(valDiv);
        dataDisplay.appendChild(item);
    }
    
    // Populate JSON view
    jsonOutput.textContent = JSON.stringify(data, null, 2);
}

toggleJsonBtn.addEventListener('click', () => {
    const isHidden = jsonOutput.style.display === 'none' || jsonOutput.style.display === '';
    jsonOutput.style.display = isHidden ? 'block' : 'none';
    toggleJsonBtn.textContent = isHidden ? 'Hide JSON' : 'Show JSON';
});
