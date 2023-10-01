import os
import cv2
from io import BytesIO
from flask import Response, send_file
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from m1.image_processing import process_images, auto_crop_and_correct

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TEMP_FOLDER'] = 'static/temp'
app.secret_key = 'your_secret_key'

# Ensure the UPLOAD_FOLDER exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ref_image = request.files['ref_image']
        scanned_image = request.files['scanned_image']

        if ref_image and scanned_image:
            # Save uploaded images to the UPLOAD_FOLDER
            ref_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(ref_image.filename))
            scanned_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(scanned_image.filename))
            ref_image.save(ref_filename)
            scanned_image.save(scanned_filename)

            # Read the uploaded images
            ref_image_cv2 = cv2.imread(ref_filename)
            scanned_image_cv2 = cv2.imread(scanned_filename)

            # Auto Crop and Correct the scanned image
            ref_image_cv2, scanned_image_cv2 = auto_crop_and_correct(ref_image_cv2, scanned_image_cv2)

            # Process the images using the machine learning code
            ref_image_cv2, scanned_image_aligned_cv2 = process_images(ref_image_cv2, scanned_image_cv2)

            # Generate a unique filename for the processed image
            processed_image_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
            cv2.imwrite(processed_image_filename, scanned_image_aligned_cv2)

            return render_template('result.html', ref_image_filename=secure_filename(ref_image.filename),
                                   processed_image='uploads/processed_image.jpg')

    return render_template('index.html')

@app.route('/download_image', methods=['GET'])
def download_image():
    # Generate a PDF version of the processed image
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')

    # Convert the image to PDF (you may need to install the fpdf library)
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.image(processed_image_path, x=10, y=10, w=190)

    # Create a unique temporary filename for the PDF
    temp_pdf_filename = os.path.join(app.config['TEMP_FOLDER'], 'processed_image.pdf')

    # Save the PDF to the temporary file
    pdf.output(temp_pdf_filename)

    # Send the PDF as a downloadable file
    return send_from_directory(app.config['TEMP_FOLDER'], 'processed_image.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
