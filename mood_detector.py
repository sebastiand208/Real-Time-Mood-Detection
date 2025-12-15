import cv2
import numpy as np
import random
import sys

def detect_edges_canny(image, sigma=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale

    # Apply Gaussian Blur to reduce noise, kernel size (5, 5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Automatic Canny threshold computation based on median intensity
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # Apply Canny Edge Detector
    edges = cv2.Canny(blurred, lower, upper)
    return edges

# Fits a polynomial using RANSAC (Random Sample Consensus)
def fit_poly_ransac(x_points, y_points, degree=2, iterations=100, threshold=2.0):
    """
    Args:
        x_points, y_points: Arrays of coordinates
        degree: Degree of polynomial (2 for parabola)
        iterations: Number of RANSAC trials
        threshold: Distance threshold to consider a point an inlier
    """

    if len(x_points) < degree + 1:
        return None

    best_model = None
    best_inliers_count = -1

    num_points = len(x_points)

    sample_size = degree + 1 # Minimum points needed to fit the model (e.g., 3 for a parabola)

    for _ in range(iterations):
        # Select random sample using random indices to pick points
        sample_indices = np.random.choice(num_points, sample_size, replace=False)
        x_sample = x_points[sample_indices]
        y_sample = y_points[sample_indices]

        try:
            # Fit model to sample (Least Squares) (Returns coefficients [a, b, c] for ax^2 + bx + c)
            coeffs = np.polyfit(x_sample, y_sample, degree)
            model_func = np.poly1d(coeffs)

            # Calculate error for all points
            y_pred = model_func(x_points)
            errors = np.abs(y_points - y_pred)

            # Count inliers
            inliers_count = np.sum(errors < threshold)

            # Keep best model
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_model = coeffs

        except np.linalg.LinAlgError:
            continue  # Skip singular matrices

    # Refit model using all inliers of the best model (Refining step)
    if best_model is not None:
        final_func = np.poly1d(best_model)
        y_pred_all = final_func(x_points)
        errors_all = np.abs(y_points - y_pred_all)
        inlier_mask = errors_all < threshold

        if np.sum(inlier_mask) > sample_size:
            best_model = np.polyfit(x_points[inlier_mask], y_points[inlier_mask], degree)

    return best_model


def main():
    # Handle camera selection
    camera_index = 1
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
            print(f"Attempting to open camera with index: {camera_index}")
        except ValueError:
            print("Invalid index provided. Using default camera (0).")

    # Initialize Webcam
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open webcam (Index {camera_index}). Try using index 0 or 1.")
        return

    # Load Haar Cascade for basic Face Detection (ROI extraction)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Mood Detector Started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw Face Rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract Mouth ROI (Region of Interest)
            mouth_roi_y = y + int(h * 0.6)
            mouth_roi_h = int(h * 0.25)
            mouth_roi_x = x + int(w * 0.2)
            mouth_roi_w = int(w * 0.6)

            mouth_roi = frame[mouth_roi_y:mouth_roi_y + mouth_roi_h, mouth_roi_x:mouth_roi_x + mouth_roi_w]

            if mouth_roi.size == 0:
                continue

            # Edge Detection (Canny)
            edges = detect_edges_canny(mouth_roi)

            # Get coordinates of edge pixels (white pixels)
            y_coords, x_coords = np.nonzero(edges)

            if len(x_coords) > 10:  # Only proceed if we have enough edge data
                # Geometric Modeling (RANSAC Polynomial Fitting) (Fit a parabola: y = ax^2 + bx + c)
                coeffs = fit_poly_ransac(x_coords, y_coords, degree=2, iterations=50, threshold=3.0)

                if coeffs is not None:
                    a, b, c = coeffs

                    '''
                    --Classification--
                    
                    In Image coordinates, Y increases downwards.
                    A smile looks like a "U" in the real world.
                    In inverted Y-axis (image), a "U" shape corresponds to a < 0.
                    A frown (upside down U) corresponds to a > 0.
                    '''

                    mood = "Neutral"
                    color = (255, 255, 0)
                    threshold_sensitivity = 0.0005  # Sensitivity tuning parameter (tau)

                    if a < -threshold_sensitivity:
                        mood = "Smiling"
                        color = (0, 255, 255)  # Yellow
                    elif a > threshold_sensitivity:
                        mood = "Frowning"
                        color = (0, 0, 255)  # Red

                    # Visualizing the fitted curve (Generate x values for plotting)
                    plot_x = np.linspace(0, mouth_roi_w, num=50)
                    plot_y = np.polyval(coeffs, plot_x)

                    # Convert to integer points for OpenCV drawing
                    points = np.array([np.stack((plot_x, plot_y), axis=1)], dtype=np.int32)
                    cv2.polylines(mouth_roi, points, isClosed=False, color=color, thickness=2)

                    # Display Mood Text on main frame
                    cv2.putText(frame, f"Mood: {mood}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Show the ROI boundaries
            cv2.rectangle(frame, (mouth_roi_x, mouth_roi_y), (mouth_roi_x + mouth_roi_w, mouth_roi_y + mouth_roi_h), (255, 0, 0), 1)

            # Show edges in a separate small window for debugging
            cv2.imshow("Mouth Edges", edges)

        cv2.imshow('Mood Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()