# Image Alignment and Processing

<img src='https://github.com/KrishGaur1354/ML-Image-Alignment/blob/main/image-aligned.png'>

The script allows users to upload a reference form and a scanned form, and it aligns the scanned form to the reference form using ~~ORB (Oriented FAST and Rotated BRIEF)~~ SIFT (Scale-Invariant Feature Transform) feature matching.

---

## Table of Content

- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following requirements installed:

- Python 3.x
- OpenCV (cv2)
- Flask
- Jupyter Notebook (for the script)
- ipywidgets and other required libraries (see script)
- SIFT (Scale-Invariant Feature Transform)

You can install the required Python libraries using `pip`. For example:

```bash
pip install opencv-python-headless flask jupyter ipywidgets
```

## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/KrishGaur1354/ML-Image-Alignment.git
   ```

2. Navigate to the repository folder:

   ```bash
   cd image-alignment
   ```

3. Run the Jupyter Notebook script to align and process images. This will open a Jupyter Notebook interface in your web browser. Follow the instructions in the notebook to upload and process images.

4. To run the Flask web application, execute the following command in your terminal:

   ```bash
   python app.py
   ```

5. Access the web application in your web browser at `http://localhost:5000`. You can upload reference and scanned images, and the application will align and display the results.

## Installation

No additional installation is required for the Jupyter Notebook script. For the Flask application, ensure you have the required Python libraries installed (see Prerequisites), and then follow the usage instructions.

## How It Works

The script and web application use OpenCV to align images based on SIFT (Scale-Invariant Feature Transform) feature matching. Here's how it works:

1. Users upload a reference form and a scanned form in the web application.

2. The script reads and processes the uploaded images using OpenCV:
   - Converts images to grayscale.
   - Detects SIFT features and computes descriptors.
   - Matches features and filters out good matches.
   - Calculates a homography matrix for alignment.
   - Warps the scanned form to align with the reference form.

3. The aligned scanned form is displayed in the web application, and users can download it.

## Contributing

Contributions are welcome! If you have any improvements, bug fixes, or feature suggestions, please open an issue or create a pull request.

Feel free to customize the README to include more details, such as screenshots, specific usage examples, or additional sections as needed.
