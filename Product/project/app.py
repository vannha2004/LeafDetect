from flask import Flask, request, jsonify, send_from_directory, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import json
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Load các mô hình và class indices ---
leaf_model = load_model('model/leaf_notLeaf/leaf_vs_notleaf_model.keras')
with open('model/leaf_notLeaf/class_indices.json') as f:
    leaf_class_names = json.load(f)

models = {}
for name, path, class_json in [
    ('cnn', 'model/cnn/best_model_cnn.keras', 'model/class_indices.json'),
    ('mobilenetv2', 'model/mobilenetv2/best_model_mobilenet.keras', 'model/class_indices.json'),
    ('resnet', 'model/resnet50/best_model_resnet50.keras', 'model/class_indices.json')
]:
    models[name] = {
        'model': load_model(path),
        'class_indices_path': class_json
    }

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"[DEBUG] Laplacian variance: {variance}")
    return variance < 300

def is_leaf(image_path, threshold=0.5):
    try:
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = leaf_model.predict(img_array)[0][0]
        predicted_label = leaf_class_names.get('0') if prediction <= threshold else leaf_class_names.get('1')
        print(f"[DEBUG] Leaf prediction: {prediction:.4f}, Predicted label: {predicted_label}")
        return predicted_label == 'leaf'
    except Exception as e:
        print(f"[DEBUG] Lỗi kiểm tra lá: {str(e)}")
        return False

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)

    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img_cv = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)

    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    img_cv = cv2.filter2D(img_cv, -1, kernel_sharpening)

    img_array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

LABEL_VIETNAMESE = {
    'dom_la': 'Đốm lá',
    'dom_vi_khuan': 'Đốm vi khuẩn',
    'healthy': 'Lá khỏe mạnh',
    'moc_suong': 'Mốc sương',
    'nam_tong_hop': 'Bệnh nấm tổng hợp',
    'phan_trang': 'Phấn trắng'
}

TREATMENT_SUGGESTIONS = {
    'dom_la': {
        'prevention_vi': 'Cắt tỉa, phân vi lượng',
        'treatment_vi': 'Chlorothalonil, Nekko 69WP'
    },
    'dom_vi_khuan': {
        'prevention_vi': 'Vệ sinh, cắt lá nhiễm',
        'treatment_vi': 'Ziflo 76WG, Dithane M45'
    },
    'healthy': {
        'prevention_vi': 'Duy trì chăm sóc thường xuyên: tưới nước đúng cách, ánh sáng và bón phân',
        'treatment_vi': 'Không cần điều trị'
    },
    'moc_suong': {
        'prevention_vi': 'Thoát nước, thông khí vườn',
        'treatment_vi': 'Ridomil Gold, Antracol, Chlorothalonil'
    },
    'nam_tong_hop': {
        'prevention_vi': 'IPM, phân bón cân đối',
        'treatment_vi': 'Thalonil, Nekko 69WP'
    },
    'phan_trang': {
        'prevention_vi': 'Sữa, baking soda, tia nắng, giống kháng',
        'treatment_vi': 'Lưu huỳnh, neem, baking soda, thuốc hữu cơ'
    }
}
@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model', 'cnn')
    if model_name not in models:
        return jsonify({'error': 'Tên mô hình không hợp lệ'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'Không có tệp ảnh được tải lên'}), 400

    file = request.files['image']
    upload_dir = os.path.join(app.static_folder, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    if not is_leaf(file_path):
        return jsonify({'error': 'Ảnh không phải là lá'}), 400

    image = cv2.imread(file_path)
    if is_blurry(image):
        return jsonify({'error': 'Ảnh quá mờ, vui lòng sử dụng ảnh rõ nét hơn.'}), 400

    # --- Tiền xử lý và dự đoán ---
    model_info = models[model_name]
    model = model_info['model']
    input_size = model.input_shape[1:3]
    img_input = preprocess_image(file_path, target_size=input_size)

    with open(model_info['class_indices_path']) as f:
        class_indices = json.load(f)
        idx_to_label = {v: k for k, v in class_indices.items()}

    predictions = model.predict(img_input)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_label = idx_to_label[predicted_index]
    confidence = float(np.max(predictions))

    label_vi = LABEL_VIETNAMESE.get(predicted_label, predicted_label)
    treatment_info = TREATMENT_SUGGESTIONS.get(predicted_label, {
        'prevention_vi': 'Không rõ',
        'treatment_vi': 'Không rõ'
    })

    probabilities = {
        LABEL_VIETNAMESE.get(idx_to_label[i], idx_to_label[i]): round(float(prob) * 100, 2)
        for i, prob in enumerate(predictions)
    }

    result_message = (
        f"Kết quả chắc chắn: {label_vi} (Độ chính xác: {round(confidence * 100, 2)}%)"
        if confidence >= 0.8
        else f"Kết quả chưa chắc chắn ({round(confidence * 100, 2)}%). Có thể là: {label_vi}"
    )

    response = {
        'label': label_vi,
        'confidence': round(confidence * 100, 2),
        'result_message': result_message,
        'probabilities': probabilities,
        'prevention': treatment_info['prevention_vi'],
        'treatment': treatment_info['treatment_vi']
    }

    if predicted_label != 'healthy':
        try:
            resized_img = cv2.resize(image, (100, 100))
            flat = resized_img.reshape((-1, 3))

            # --- MeanShift clustering ---
            from sklearn.cluster import MeanShift, estimate_bandwidth
            bandwidth = estimate_bandwidth(flat, quantile=0.1, n_samples=500)

            if bandwidth <= 0:
                print("[DEBUG] Bandwidth quá nhỏ, bỏ qua.")
            else:
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(flat)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_

                # --- Tạo ảnh phân cụm ---
                segmented_img = cluster_centers[labels].reshape(resized_img.shape).astype(np.uint8)
                segmented_gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

                # --- Black == Diseases ---
                _, mask = cv2.threshold(segmented_gray, 100, 255, cv2.THRESH_BINARY_INV)

                # --- Tìm contour ---
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w_box, h_box = cv2.boundingRect(largest)

                    # Scale bbox từ ảnh nhỏ về ảnh gốc
                    scale_x = image.shape[1] / 100.0
                    scale_y = image.shape[0] / 100.0

                    response['bounding_box'] = [
                        int(x * scale_x), int(y * scale_y),
                        int(w_box * scale_x), int(h_box * scale_y)
                    ]

                    print(f"[DEBUG] BBox từ MeanShift: {response['bounding_box']}")
                else:
                    print("[DEBUG] Không tìm được vùng bệnh từ phân cụm.")
        except Exception as e:
            print(f"[DEBUG] Lỗi MeanShift phát hiện bệnh: {e}")
    return jsonify(response)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.static_folder, 'uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)
