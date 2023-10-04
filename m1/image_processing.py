import cv2
import numpy as np

def auto_crop_and_correct(ref_image, scanned_image):
    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
    scanned_gray = cv2.cvtColor(scanned_image, cv2.COLOR_RGB2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(ref_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(scanned_gray, None)

    # Create a brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Choose the top N matches (you can adjust N as needed)
    N = 200
    good_matches = matches[:N]

    # Extract the matched keypoints
    src_pts = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Use the homography matrix to warp the scanned image
    corrected_image = cv2.warpPerspective(scanned_image, M, (ref_image.shape[1], ref_image.shape[0]))

    return ref_image, corrected_image

def process_images(ref_image, scanned_image):
    # ACC (Automatic Crop and Correct)
    ref_image, scanned_image = auto_crop_and_correct(ref_image, scanned_image)

    # Calculate the pixel size of the original reference and scanned images
    ref_height, ref_width, _ = ref_image.shape
    scanned_height, scanned_width, _ = scanned_image.shape

    print(f"Reference Image Size (pixels): {ref_width} x {ref_height}")
    print(f"Scanned Image Size (pixels): {scanned_width} x {scanned_height}")

    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
    scanned_gray = cv2.cvtColor(scanned_image, cv2.COLOR_RGB2GRAY)

    # Detect SIFT features and compute descriptors
    MAX_NUM_FEATURES = 10000  # Increase the number of features
    sift = cv2.SIFT_create(MAX_NUM_FEATURES)
    keypoints1, descriptors1 = sift.detectAndCompute(ref_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(scanned_gray, None)

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = list(matcher.match(descriptors1, descriptors2, None))

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not-so-good matches
    numGoodMatches = int(len(matches) * 0.5)
    matches = matches[:numGoodMatches]

    # Calculate and display the percentage and number of feature matches
    total_matches = len(matches)
    total_keypoints1 = len(keypoints1)
    total_keypoints2 = len(keypoints2)
    match_percentage = (total_matches / min(total_keypoints1, total_keypoints2)) * 100

    print(f"Number of feature matches: {total_matches}")
    print(f"Percentage of feature matches: {match_percentage:.2f}%")

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to warp image
    height, width, channels = ref_image.shape
    scanned_image_aligned = cv2.warpPerspective(scanned_image, h, (width, height))

    # Calculate the pixel size of the aligned scanned image
    aligned_height, aligned_width, _ = scanned_image_aligned.shape
    print(f"Aligned Scanned Image Size (pixels): {aligned_width} x {aligned_height}")

    return ref_image, scanned_image_aligned
