from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/paper')
def paper():
    return render_template('paper.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        result_path = process_image(file_path, request.form.get('doc_type'))
        return jsonify({'result_url': f'/processed/{os.path.basename(result_path)}'})
    return redirect(url_for('index'))

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

def process_image(image_path, mode):
    image = cv2.imread(image_path)
    
    if mode == 'paper':
        result_image = process_paper_document(image)
    else:
        result_image = process_e_document(image)
    
    result_path = os.path.join(PROCESSED_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, result_image)
    return result_path

def process_e_document(image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(gray_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    line_min_width = 10
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    img_bin_final = img_bin_h | img_bin_v
    final_kernel = np.ones((3, 3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    
    for x, y, w, h, area in stats[2:]:
        aspect_ratio = w / h
        if 20 <= w <= 80 and 20 <= h <= 80 and 0.8 <= aspect_ratio <= 1.2:
            roi = gray_scale[y:y+h, x:x+w]
            _, checkbox_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            total_pixels = checkbox_thresh.size
            non_zero_pixels = cv2.countNonZero(checkbox_thresh)
            fill_ratio = non_zero_pixels / total_pixels
            if fill_ratio > 0.5:
                cv2.putText(image, 'Yes', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, bottomLeftOrigin=False)
            else:
                cv2.putText(image, 'No', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image

def process_paper_document(image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(gray_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    line_min_width = 3
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    img_bin_final = img_bin_h | img_bin_v
    final_kernel = np.ones((1, 1), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)
    contours_first, hierarchy_first = cv2.findContours(img_bin_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    original_image = image.copy()
    canny_image = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), (5, 5), 1), 10, 50)
    contours_second, hierarchy_second = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    rectangle_vertices_list = []
    for i in contours_first:
        for j in contours_second:
            x, y, width, height = cv2.boundingRect(i)
            if (width > 10 and width < 85) and (height > 10 and height < 85):
                param_lenght = cv2.arcLength(i, True)
                param_approx = cv2.approxPolyDP(i, 0.02 * param_lenght, True)
                if len(param_approx) == 4:
                    rectangle_vertices_list.append(param_approx)
    for i in rectangle_vertices_list:
        cv2.drawContours(image, i, -1, (0, 255, 0), 2)
        temp = i
        image_area = canny_image[i[0][0][1]:i[2][0][1] + 2, i[0][0][0]:i[3][0][0] + 1]
        if cv2.countNonZero(image_area) > cv2.contourArea(i) * 0.5:
            if i[0][0][0] - 1 <= i[2][0][0] and i[0][0][1] - 1 <= i[2][0][1]:
                cv2.putText(image, 'Yes', i[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2, bottomLeftOrigin=False)
            else:
                cv2.putText(image, 'Yes', i[1][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2, bottomLeftOrigin=False)
        else:
            if i[0][0][0] - 1 <= i[2][0][0] and i[0][0][1] - 1 <= i[2][0][1]:
                cv2.putText(image, 'No', i[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2, bottomLeftOrigin=False)
            else:
                cv2.putText(image, 'No', i[1][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2, bottomLeftOrigin=False)
    return image

if __name__ == "__main__":
    app.run(debug=True)
