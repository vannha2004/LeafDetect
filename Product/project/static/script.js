$(document).ready(function() {
    // --- 1. Initialize variables ---
    let chart = null;
    let history = JSON.parse(localStorage.getItem('diseaseHistory')) || [];
    
    // --- 2. DOM elements ---
    const elements = {
        imageUpload: $('#imageUpload'),
        classifyBtn: $('#classifyBtn'),
        clearHistoryBtn: $('#clearHistoryBtn'),
        previewImage: $('#previewImage'),
        diseaseCanvas: $('#diseaseCanvas'),
        predictionText: $('#predictionText'),
        treatmentCards: $('#treatmentCards'),
        preventionVi: $('#preventionVi'),
        treatmentVi: $('#treatmentVi'),
        warningMessage: $('#warningMessage'),
        chartWrapper: $('#chartWrapper'),
        historyBody: $('#historyBody'),
        modelSelect: $('#modelSelect'),
        uploadArea: $('#uploadArea'),
        imagePreviewWrapper: $('#imagePreviewWrapper'),
        noImagePlaceholder: $('#noImagePlaceholder'),
        resultPlaceholder: $('#resultPlaceholder'),
        resultContent: $('#resultContent')
    };
    
    // --- 3. Initialize history display ---
    updateHistoryDisplay();
    
    // --- 4. File upload handling ---
    elements.imageUpload.on('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                elements.previewImage.attr('src', e.target.result).removeClass('d-none');
                elements.noImagePlaceholder.addClass('d-none');
                elements.uploadArea.removeClass('pulse-animation').addClass('active');
                
                // Reset display
                elements.diseaseCanvas.hide();
                elements.diseaseCanvas[0].getContext('2d').clearRect(0, 0, 
                    elements.diseaseCanvas[0].width, elements.diseaseCanvas[0].height);
                    
                elements.chartWrapper.hide();
                if (chart) {
                    chart.destroy();
                    chart = null;
                }
                
                elements.warningMessage.addClass('d-none').text('');
                elements.resultPlaceholder.show();
                elements.resultContent.hide();
                
                elements.classifyBtn.prop('disabled', false);
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Drag and drop functionality
    elements.uploadArea.on('dragover', function(e) {
        e.preventDefault();
        $(this).addClass('border-primary bg-light');
    }).on('dragleave', function() {
        $(this).removeClass('border-primary bg-light');
    }).on('drop', function(e) {
        e.preventDefault();
        $(this).removeClass('border-primary bg-light');
        if (e.originalEvent.dataTransfer.files.length) {
            elements.imageUpload[0].files = e.originalEvent.dataTransfer.files;
            elements.imageUpload.trigger('change');
        }
    });
    
    // --- 5. Classify button handler ---
    elements.classifyBtn.on('click', function() {
        const file = elements.imageUpload[0].files[0];
        if (!file) {
            showWarning('Vui lòng tải lên hình ảnh trước khi phân tích');
            return;
        }
        
        const model = elements.modelSelect.val();
        const formData = new FormData();
        formData.append('image', file);
        formData.append('model', model);
        
        // UI updates
        elements.resultPlaceholder.hide();
        elements.resultContent.show().hide().fadeIn(300);
        elements.predictionText.text('Đang phân tích...');
        elements.classifyBtn.prop('disabled', true);
        elements.treatmentCards.hide();
        elements.warningMessage.addClass('d-none').text('');
        elements.chartWrapper.hide();
        
        if (chart) {
            chart.destroy();
            chart = null;
        }
        
        // AJAX request
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                if (data.error) {
                    showWarning(data.error);
                    elements.predictionText.text('Phân tích thất bại');
                    return;
                }
                
                // Update UI with results
                elements.predictionText.text(data.label || 'Không xác định');
                
                if (data.blur_warning) {
                    showWarning(data.blur_warning);
                }
                
                if (data.confidence && data.confidence < 70) {
                    const warning = `Độ tin cậy thấp: ${data.confidence.toFixed(2)}%. Vui lòng kiểm tra lại hình ảnh.`;
                    showWarning(warning);
                }
                
                elements.preventionVi.text(data.prevention || 'Không có thông tin phòng ngừa');
                elements.treatmentVi.text(data.treatment || 'Không có thông tin điều trị');
                elements.treatmentCards.fadeIn(300);
                
                // Draw chart if probabilities exist
                if (data.probabilities) {
                    const labels = Object.keys(data.probabilities);
                    const values = Object.values(data.probabilities);
                    
                    if (chart) chart.destroy();
                    
                    const ctx = document.getElementById('predictionChart').getContext('2d');
                    chart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Tỉ lệ dự đoán (%)',
                                data: values.map(v => parseFloat(v.toFixed(2))),
                                backgroundColor: 'rgba(75, 192, 192, 0.7)',
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            indexAxis: 'y',
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    beginAtZero: true,
                                    max: 100,
                                    ticks: {
                                        callback: function(value) {
                                            return value + '%';
                                        }
                                    }
                                }
                            },
                            plugins: {
                                legend: {
                                    display: false
                                }
                            }
                        }
                    });
                    elements.chartWrapper.fadeIn(300);
                }
                
                // Draw bounding box if exists
                if (data.bounding_box) {
                    const ctx = elements.diseaseCanvas[0].getContext('2d');
                    const img = new Image();
                    img.src = elements.previewImage.attr('src');
                    
                    img.onload = function() {
                        const displayWidth = elements.previewImage.width();
                        const displayHeight = elements.previewImage.height();
                        
                        elements.diseaseCanvas[0].width = displayWidth;
                        elements.diseaseCanvas[0].height = displayHeight;
                        elements.diseaseCanvas.show();
                        
                        ctx.clearRect(0, 0, displayWidth, displayHeight);
                        ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
                        
                        // Draw bounding box
                        ctx.strokeStyle = 'red';
                        ctx.lineWidth = 4;
                        
                        const originalWidth = img.naturalWidth;
                        const originalHeight = img.naturalHeight;
                        const scaleX = displayWidth / originalWidth;
                        const scaleY = displayHeight / originalHeight;
                        
                        const bbox = data.bounding_box;
                        const x = bbox[0] * scaleX;
                        const y = bbox[1] * scaleY;
                        const width = bbox[2] * scaleX;
                        const height = bbox[3] * scaleY;
                        
                        ctx.strokeRect(x, y, width, height);
                    };
                }
                
                // Add to history
                addToHistory({
                    date: new Date().toLocaleString(),
                    image: elements.previewImage.attr('src'),
                    prediction: data.label,
                    prevention: data.prevention || '',
                    treatment: data.treatment || ''
                });
            },
            error: function(xhr) {
                let errorMsg = 'Đã xảy ra lỗi khi gửi yêu cầu phân tích';
                try {
                    const errorData = JSON.parse(xhr.responseText);
                    errorMsg = errorData.error || errorMsg;
                } catch (e) {
                    console.error(e);
                }
                
                showWarning(errorMsg);
                elements.predictionText.text('Phân tích thất bại');
            },
            complete: function() {
                elements.classifyBtn.prop('disabled', false);
            }
        });
    });
    
    // --- 6. Clear history handler ---
    elements.clearHistoryBtn.on('click', function() {
        if (confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử phân tích? Hành động này không thể hoàn tác.')) {
            history = [];
            localStorage.setItem('diseaseHistory', JSON.stringify(history));
            updateHistoryDisplay();
            
            // Show placeholder
            elements.historyBody.html(`
                <tr>
                    <td colspan="5" class="text-center text-muted py-4">
                        <i class="bi bi-inbox fs-1 d-block mb-2"></i>
                        Chưa có lịch sử phân tích nào
                    </td>
                </tr>
            `);
        }
    });
    
    // --- Helper functions ---
    function showWarning(message) {
        elements.warningMessage.text(message).removeClass('d-none').addClass('show');
    }
    
    function addToHistory(item) {
        history.push(item);
        
        // Limit history to last 10 items
        if (history.length > 10) {
            history = history.slice(-10);
        }
        
        localStorage.setItem('diseaseHistory', JSON.stringify(history));
        updateHistoryDisplay();
    }
    
    function updateHistoryDisplay() {
        if (history.length === 0) return;
        
        let html = '';
        history.forEach(item => {
            html += `
                <tr>
                    <td>${item.date}</td>
                    <td><img src="${item.image}" class="history-img" alt="Ảnh lịch sử"></td>
                    <td>${item.prediction}</td>
                    <td>${item.prevention || ''}</td>
                    <td>${item.treatment || ''}</td>
                </tr>
            `;
        });
        
        elements.historyBody.html(html);
    }
});
