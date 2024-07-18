from flask import Flask, request, render_template, send_from_directory, jsonify, redirect, url_for
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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
    # Binarize the image using adaptive thresholding and invert it
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(gray_scale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


    # Define the minimum checkbox size
    line_min_width = 3

    # Define vertical and horizontal kernels for morphological operations
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)

    # Detect horizontal lines
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)

    # Detect vertical lines
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)


    # Merge horizontal and vertical lines
    img_bin_final = img_bin_h | img_bin_v

    # Apply dilation to close small gaps
    final_kernel = np.ones((1, 1), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)


    contours_first, hierarchy_first = cv2.findContours(img_bin_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    original_image = image.copy()
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(grayscale_image, (5, 5), 1)
    canny_image = cv2.Canny(blur_image, 10, 50)

    copy_image = original_image.copy()
    draw_what_you_need_image = original_image.copy()
    #finished_all_lines_image = original_image.copy()

    contours_second, hierarchy_second = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    rectangle_list_first = []
    rectangle_vertices_list_first = []
    for i in contours_first:
        x, y, width, height = cv2.boundingRect(i)

        if (width > 10 and width < 85) and (height > 10 and height < 85):
            param_lenght = cv2.arcLength(i, True)
            param_approx = cv2.approxPolyDP(i, 0.02 * param_lenght, True)
            if len(param_approx) == 4: #or len(param_approx) == 5:
                rectangle_list_first.append(i)
                rectangle_vertices_list_first.append(param_approx)

    rectangle_list_second = []
    rectangle_vertices_list_second = []
    # for j in contours_first:
    for i in contours_second:
        x, y, width, height = cv2.boundingRect(i)

        if (width > 10 and width < 85) and (height > 10 and height < 85):
            param_lenght = cv2.arcLength(i, True)
            param_approx = cv2.approxPolyDP(i, 0.02 * param_lenght, True)
            if len(param_approx) == 4: 
                rectangle_list_second.append(i)
                rectangle_vertices_list_second.append(param_approx)

    contours = contours_second + contours_first
    # Flatten contours and find unique ones
    flat_contours = [c.reshape(-1, 2) for c in contours]

    unique_contours = []
    seen = set()

    for contour in flat_contours:
        contour_tuple = tuple(map(tuple, contour))  # Convert to a tuple of tuples for hashing
        if contour_tuple not in seen:
            seen.add(contour_tuple)
            unique_contours.append(contour)

    # Convert back to original shape
    unique_contours = [np.expand_dims(c, axis=1) for c in unique_contours]

    rectangle_list = rectangle_list_first + rectangle_list_second
    rectangle_vertices_list = rectangle_vertices_list_first + rectangle_vertices_list_second

    # Flatten contours and find unique ones
    flat_rectangle_list = [c.reshape(-1, 2) for c in rectangle_vertices_list]

    unique_rectangle_list = []
    seen = set()

    for rectangle in flat_rectangle_list:
        rectangle_tuple = tuple(map(tuple, rectangle))  # Convert to a tuple of tuples for hashing
        if rectangle_tuple not in seen:
            seen.add(rectangle_tuple)
            unique_rectangle_list.append(rectangle)

    # Convert back to original shape
    unique_rectangle_list = [np.expand_dims(c, axis=1) for c in unique_rectangle_list]

    unique_rectangle_list.sort(key=lambda c: c[0][0][0])  # Sorting by x-coordinate of the first point

    image_area = grayscale_image


    prev = np.array([[[1, 1]], [[1, 1]], [[1, 1]], [[1, 1]]])
    for i in unique_rectangle_list:
        cv2.drawContours(image, i, -1, (0, 255, 0), 2)
        temp = i
        image_area = canny_image[i[0][0][1]:i[2][0][1] + 2, i[0][0][0]:i[3][0][0] + 1]
        if (abs(prev[0][0][1] - i[0][0][1]) < 1 or abs(prev[2][0][1] - i[2][0][1]) < 3 or abs(prev[1][0][1] - i[1][0][1]) < 5):
            continue
        else: 
            if abs(prev[1][0][0] - i[1][0][0]) < 3 and not(abs(prev[1][0][1] - i[1][0][1]) > 5):
                continue
            if not(abs(prev[3][0][1] - i[3][0][1]) > 1) and abs(prev[3][0][0] - i[3][0][0]) < 15:
                continue
        if cv2.countNonZero(image_area) > cv2.contourArea(i) * 0.4:
            prev = i
            if i[0][0][0] - 1 <= i[2][0][0] and i[0][0][1] - 1 <= i[2][0][1]:
                cv2.putText(image, 'Yes', i[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, bottomLeftOrigin=False)
            else:
                cv2.putText(image, 'Yes', i[1][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, bottomLeftOrigin=False)
        else:
            prev = i
            if i[0][0][0] - 1 <= i[2][0][0] and i[0][0][1] - 1 <= i[2][0][1]:
                cv2.putText(image, 'No', i[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, bottomLeftOrigin=False)
            else:
                cv2.putText(image, 'No', i[1][0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, bottomLeftOrigin=False)
    # Invert the image and find the connected components
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    # Visualize the connected component image


    # Loop through each detected component and draw a bounding box
    for x, y, w, h, area in stats[2:]:
        aspect_ratio = w / h
        if 20 <= w <= 80 and 20 <= h <= 80 and 0.8 <= aspect_ratio <= 1.2:  # Adjusted size and aspect ratio check
            # Extract the region of interest
            roi = gray_scale[y:y+h, x:x+w]
            # Threshold the ROI
            _, checkbox_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Check if the checkbox is checked based on pixel density within the bounding box
            total_pixels = checkbox_thresh.size
            non_zero_pixels = cv2.countNonZero(checkbox_thresh)
            fill_ratio = non_zero_pixels / total_pixels
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

if __name__ == "__main__":
    app.run(debug=True)
