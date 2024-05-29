import os
import cv2
import numpy as np
from scipy.spatial import distance
import group_project.AJBastroalign as aa
from skimage.measure import ransac
import astroalign as aa
from skimage.transform import SimilarityTransform

class ImageProcessor:
    def __init__(self, directory='group15'):
        self.directory = directory

    def list_image_paths(self):
        home_directory = os.path.expanduser('~')
        img_directory = os.path.join(home_directory, self.directory)
        images = [f for f in os.listdir(img_directory) if os.path.isfile(os.path.join(img_directory, f))]
        
        image_earth_path = None
        image_moon_path = None

        for img in images:
            if img == 'viewEarth.png':
                image_earth_path = os.path.join(img_directory, img)
            elif img == 'viewMoon.png':
                image_moon_path = os.path.join(img_directory, img)
        
        return image_earth_path, image_moon_path

    def get_image_paths(self):
        return self.list_image_paths()
    
    def blend_images(self, image_earth, image_moon):
    
        # Convert images to grayscale
        gray_earth = cv2.cvtColor(image_earth, cv2.COLOR_BGR2GRAY)
        gray_moon = cv2.cvtColor(image_moon, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        r_keypoints, r_descriptors = sift.detectAndCompute(gray_earth, None)
        l_keypoints, l_descriptors = sift.detectAndCompute(gray_moon, None)

        # Convert keypoints to numpy arrays
        source = np.array([[p.pt[0], p.pt[1]] for p in r_keypoints], dtype=np.float64).reshape(-1, 2)
        target = np.array([[p.pt[0], p.pt[1]] for p in l_keypoints], dtype=np.float64).reshape(-1, 2)

        # Create a FLANN matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match descriptors using FLANN
        matches = flann.knnMatch(r_descriptors, l_descriptors, k=2)

        # Filter matches using ratio test
        good_matches = []
        for m, n in matches:  # Iterate over three nearest neighbors
            if m.distance < 0.7 * n.distance:  # Use ratio test
                good_matches.append(m)

        # Extract matched keypoints
        matched_r_keypoints = source[[m.queryIdx for m in good_matches]]
        matched_l_keypoints = target[[m.trainIdx for m in good_matches]]

        # Convert matched keypoints to homogeneous coordinates
        matched_r_keypoints_homogeneous = np.hstack((matched_r_keypoints, np.ones((len(matched_r_keypoints), 1))))
        matched_l_keypoints_homogeneous = np.hstack((matched_l_keypoints, np.ones((len(matched_l_keypoints), 1))))

        # Define RANSAC model (Affine transformation)
        model, inliers = ransac((matched_r_keypoints_homogeneous, matched_l_keypoints_homogeneous),
                            SimilarityTransform, min_samples=5, residual_threshold=5, max_trials=10000)

        inlier_matched_r_keypoints = np.array(matched_r_keypoints)[inliers]
        inlier_matched_l_keypoints = np.array(matched_l_keypoints)[inliers]

        r_keypoint_array_filtered = np.array(inlier_matched_r_keypoints)
        l_keypoint_array_filtered = np.array(inlier_matched_l_keypoints)
        
        r_keypoint_array_filtered = np.unique(r_keypoint_array_filtered, axis=0)
        l_keypoint_array_filtered = np.unique(l_keypoint_array_filtered, axis=0)
        # Check if any keypoints were filtered out
        if len(r_keypoint_array_filtered) == 0 or len(l_keypoint_array_filtered) == 0:
            print("Warning: No valid keypoints found. Ensure the images have enough features.")
        else:
            # Find the transformation matrix using astroalign.find_transform
            try:
                transf, (src_pts, dst_pts) = aa.find_transform(r_keypoint_array_filtered, l_keypoint_array_filtered)
                print("Transformation matrix found successfully.")
                
                similarity_transf = SimilarityTransform()
                similarity_transf.estimate(src_pts, dst_pts)
       
                # Convert transform to form OpenCV can use
                homography = np.matrix(transf, np.float32)

                # Apply transform to right image
                result = cv2.warpPerspective(image_moon, homography, (image_earth.shape[1]+400, image_earth.shape[0]),flags=cv2.INTER_LINEAR)

                # Blending the warped image with the second image using alpha blending
                padded_left_img = cv2.copyMakeBorder(image_earth, 0, 0, 0, result.shape[1] - image_earth.shape[1],cv2.BORDER_CONSTANT )
                alpha = 0.5  # blending factor
                blended_image = cv2.addWeighted(padded_left_img, alpha, result, 1 - alpha, 0)

                # Display the blended image
                # Convert BGR image to RGB (OpenCV reads images in BGR format)
                blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
                return blended_image_rgb

            except ValueError as e:
                print(f"Failed to find transformation matrix: {e}")
        
    
    def is_circle(self, contour, threshold=0.4):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        return circularity > threshold

    def detect_earth(self, image, color_lower=np.array([100, 50, 50]), color_upper=np.array([140, 255, 255]), min_radius=5):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_lower, color_upper)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        processed_image = image.copy()
        best_circle = None
        max_radius = 0

        for contour in contours:
            if self.is_circle(contour):
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                if radius > min_radius and radius > max_radius:
                    max_radius = radius
                    best_circle = (center, radius)
                    diameter = radius * 2

        height, _, _ = image.shape

        if best_circle:
            center, radius = best_circle
            cv2.circle(processed_image, center, radius, (0, 255, 0), 4)
            cv2.line(processed_image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (255, 0, 0), 2)
            return processed_image, height, diameter

        return processed_image, height, None

    def detect_moon(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = image.copy()
        circle_details = []
        max_radius = 0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Adjusted HoughCircles parameters
            circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                    param1=50, param2=30, minRadius=10, maxRadius=100)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, r) in circles:
                    center_x = x + cx
                    center_y = y + cy
                    circle_details.append((center_x, center_y, r))
                    cv2.circle(roi, (cx, cy), r, (0, 255, 0), 4)
                    cv2.circle(output_image, (center_x, center_y), r, (0, 255, 0), 4)
                    if r > max_radius:
                        max_radius = r

        moon_pixel_diameter = max_radius * 2  # Diameter is twice the radius
        return output_image, moon_pixel_diameter
    
    def detect_earth_stitch(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_lower = np.array([90, 50, 50])
        color_upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, color_lower, color_upper)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        processed_image = image.copy()
        best_circle = None
        max_radius = 0

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > max_radius:
                max_radius = radius
                best_circle = (center, radius)

        if best_circle:
            center, radius = best_circle
            cv2.circle(processed_image, center, radius, (0, 255, 0), 4)
            cv2.line(processed_image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (255, 0, 0), 2)
            return processed_image, center, radius * 2  # Return the processed image, center, and diameter

        return processed_image, None, None  # No circle detected
    
    def detect_moon_stitch(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = image.copy()
        circle_details = []
        max_radius = 0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Adjusted HoughCircles parameters
            circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                    param1=50, param2=30, minRadius=1, maxRadius=20)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, r) in circles:
                    center_x = x + cx
                    center_y = y + cy
                    circle_details.append((center_x, center_y, r))
                    cv2.circle(roi, (cx, cy), r, (0, 255, 0), 4)
                    cv2.circle(output_image, (center_x, center_y), r, (0, 255, 0), 4)
                    if r > max_radius:
                        max_radius = r

        moon_pixel_diameter = max_radius * 2  # Diameter is twice the radius
        if circle_details:
            moon_center = circle_details[0][:2]
        else:
            moon_center = None

        return output_image, moon_center, moon_pixel_diameter

    def calculate_distance_earth(self, height, diameter):
        scaling_factor = 3  
        real_diameter = 12742  
        if diameter is None:
            return None
        distance = scaling_factor * (real_diameter * height / diameter)
        return int(np.ceil(distance))

    def calculate_distance_moon(self, height, diameter):
        scaling_factor = 3
        real_diameter = 3475
        if diameter is None:
            return None
        distance = scaling_factor * (real_diameter * height / diameter)
        return int(np.ceil(distance))

    def calculate_distance_stitch(self, earth_center, moon_center, earth_pixel_diameter):
        earth_real_diameter = 12742
        if earth_center is None or moon_center is None or earth_pixel_diameter == 0:
            return None
        
        # Example calculation (adjust as needed based on actual distance formula)
        min_distance = np.linalg.norm(np.array(earth_center) - np.array(moon_center))
        distance = min_distance * (earth_real_diameter / earth_pixel_diameter)
        return int(np.ceil(distance))


    def process_images(self):
        earth_path, moon_path = self.get_image_paths()
        
        # print(earth_path)

        image_earth = cv2.imread(earth_path)
        image_moon = cv2.imread(moon_path)
        # print(np.shape(image_earth))
        # cv2.imshow('earth', image_earth)
        # cv2.imshow('moon', image_moon)
        # input()

        if image_earth is None or image_moon is None:
            print("Error: Unable to load one or both images.")
            return

        blended_image = self.blend_images(image_earth, image_moon)
        
        home_directory = os.path.expanduser('~')
        group15_directory = os.path.join(home_directory, 'group15')
        
        if not os.path.exists(group15_directory):
            os.makedirs(group15_directory)
            
        panorama_path = os.path.join(group15_directory, 'panorama.png')
        cv2.imwrite(panorama_path, blended_image)
        
        measurement_file_path = os.path.join(group15_directory, 'measurement.txt')
        with open(measurement_file_path, 'w') as file:
            processed_earth_image, height, diameter = self.detect_earth(image_earth)
            earth_distance = self.calculate_distance_earth(height, diameter)
            if processed_earth_image is not None:
                file.write(f"Earth: {earth_distance} km\n")
            else:
                file.write("Earth not detected\n")
                print("Earth not detected")

            # squares = self.detect_squares(image_moon)
            # processed_moon_image, height, diameter = self.detect_circles_within_squares(image_moon, squares)
            processed_moon_image, moon_pixel_diameter = self.detect_moon(image_moon)
            moon_distance = self.calculate_distance_moon(height, moon_pixel_diameter)
            if processed_moon_image is not None:
                file.write(f"Moon: {moon_distance} km\n")
            else:
                file.write("Moon not detected\n")
                print("Moon not detected")

            
            if blended_image is not None:
                processed_image, earth_center, earth_diameter = self.detect_earth_stitch(blended_image)

                # Detect Moon
                processed_image, moon_center, moon_pixel_diameter = self.detect_moon_stitch(processed_image)

                if earth_center and moon_center:
                    # Calculate distance between Earth and Moon
                    calculated_distance = self.calculate_distance_stitch(earth_center, moon_center, earth_diameter)
                    file.write(f'Distance between Earth and Moon is: {calculated_distance} km')

        print(f"All measurements have been written to {measurement_file_path}")

# Usage
if __name__ == "__main__":
    processor = ImageProcessor()
    processor.process_images()
