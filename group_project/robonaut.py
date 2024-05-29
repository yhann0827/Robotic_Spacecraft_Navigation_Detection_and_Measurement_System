import threading
import time
import tf2_ros 
import numpy as np
import cv2
import signal
import rclpy
import math
import random
import os
from group_project import coordinates
from rclpy.node import Node
from rclpy.exceptions import ROSInterruptException
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from math import sin, cos, pi
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rclpy.clock import Clock
from ultralytics import YOLO
from group_project.stitchAndMeasure import ImageProcessor 
class RoboNaut(Node):
    def __init__(self):
        super().__init__('robotnaut')
        
        self.declare_parameter('coordinates_file_path', '')
        coordinates_file_path = self.get_parameter('coordinates_file_path').get_parameter_value().string_value
        self.coordinates = coordinates.get_module_coordinates(coordinates_file_path)
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.cv_bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.launch_image_processing, 10)
        self.subscription  # prevent unused variable warning
        
        self.laser_subscription = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.subscription  # prevent unused variable warning
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(20)
        
        self.threads = []
        
        # Initialise any flags that signal a colour has been detected (default to false)
        self.detect_colour_green = False
        self.detect_colour_red = False
        
        # Flag to indicate if a circle is found
        self.found_button = False
        self.at_center = False
        self.at_entrance = False
        self.current_module = None 
        self.start_color_detection = False
        
        self.lock = threading.Lock()
        self.last_processed_time = self.get_clock().now()
        
        self.obstacle_detected = False
        self.edge_detected = False
        self.laser_data = None

        self.closest_distance = float('inf')
        
        self.windowmodel = YOLO('/uolstore/home/users/sc22cyg/ros2_ws/src/group-project-group-15/group_project/Object_Detection_Model/WindowDetection/best.pt')
        self.planetmodel = YOLO('/uolstore/home/users/sc22cyg/ros2_ws/src/group-project-group-15/group_project/Object_Detection_Model/PlanetDetection/best.pt')
        self.detected_planets = set() 
        self.window_counter = 1
        
        self.start_window_detection = False
        self.move_closer = False
        self.contour_detection = False
        
        entrance_y = self.coordinates.module_1.entrance.y if self.current_module == 'module_1' else self.coordinates.module_2.entrance.y
        self.y_boundary = max(entrance_y, entrance_y + 2.7)
        # self.y_boundary = max(entrance_y, entrance_y + 3.2)
        # self.y_boundary = 0.173
        self.reached_boundary = False
        
        self.window_detection_event = threading.Event()
        self.window_thread_started = False

    
    def start_window_detection_thread(self):
        if not self.window_thread_started:
            window_thread = threading.Thread(target=self.detect_window_thread, daemon=True)
            window_thread.start()
            self.window_thread_started = True

    def start_walking_thread(self):
        walking_thread = threading.Thread(target=self.walk_forward_until_obstacle, daemon=True)
        walking_thread.start()

    def detect_window_thread(self):
        while not self.window_detection_event.is_set():
            time.sleep(0.1)

        while rclpy.ok() and self.start_window_detection:
            with self.lock:
                self.detect_window()  
          
    # -------------------------------------------------------------- NAVIGATION --------------------------------------------------------- #   
    def navigate_to(self, x, y, yaw):
        velocity_msg = Twist()
        velocity_msg.linear.x = 0.5 
        self.publisher.publish(velocity_msg)
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        # self.action_client.wait_for_server()
        # self.action_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)
        self.action_client.wait_for_server()
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)
        
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        if result:
            if self.is_at_center_goal():
                self.get_logger().info('Reached center of the module.')
                self.at_center = True
                # self.walk_forward_until_obstacle()
                self.start_window_detection_thread()  # Start the window detection thread
                self.start_walking_thread()  # Start the walking threa
            else:
                self.get_logger().info('At entrance, ready to start color detection.')
                self.start_color_detection = True
                self.rotate_to_detect_button()
        else:
            self.get_logger().info('Navigation failed.') 
            
    def get_current_position(self):
        try:
            self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(), timeout=rclpy.time.Duration(seconds=10))
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(seconds=0))
            return (trans.transform.translation.x, trans.transform.translation.y)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Error in getting transform: {str(e)}")
            return None
    
    
    def navigate_to_nearest_entrance(self):
        current_position = self.get_current_position()
        if current_position:
            entrance_1 = self.coordinates.module_1.entrance.x, self.coordinates.module_1.entrance.y
            entrance_2 = self.coordinates.module_2.entrance.x, self.coordinates.module_2.entrance.y
            distance_to_entrance_1 = np.linalg.norm(np.array(current_position) - np.array(entrance_1))
            distance_to_entrance_2 = np.linalg.norm(np.array(current_position) - np.array(entrance_2))

            if distance_to_entrance_1 < distance_to_entrance_2:
                self.current_module = 'module_1'
                self.at_entrance = True
                self.at_center = False 
                self.navigate_to(*entrance_1, 0)
            else:
                self.current_module = 'module_2'
                self.at_entrance = True
                self.at_center = False 
                self.navigate_to(*entrance_2, 0)    
                
    def navigate_to_center(self, module_name):
        if module_name == 'module_1':
            center_x, center_y = self.coordinates.module_1.center.x, self.coordinates.module_1.center.y
        elif module_name == 'module_2':
            center_x, center_y = self.coordinates.module_2.center.x, self.coordinates.module_2.center.y  
            
        if center_x is not None and center_y is not None:
            self.navigate_to(center_x, center_y, -1.0) 
            # self.navigate_to(center_x, center_y, -5.0) 
            
    def is_at_center_goal(self):
        current_position = self.get_current_position()
        if not current_position:
            return False

        center_1_x = self.coordinates.module_1.center.x
        center_1_y = self.coordinates.module_1.center.y
        center_2_x = self.coordinates.module_2.center.x
        center_2_y = self.coordinates.module_2.center.y

        return (
            (abs(current_position[0] - center_1_x) < 0.1 and abs(current_position[1] - center_1_y) < 0.1) or
            (abs(current_position[0] - center_2_x) < 0.1 and abs(current_position[1] - center_2_y) < 0.1)
        )                               
            
    # ----------------------------------------------------------------------------------------------------------------------------------- # 
    
             
    # -------------------------------------------------------- COLOR DETECTION ---------------------------------------------------------- # 
    def launch_image_processing(self, data):
        self.latest_image_data = data
        if not self.start_color_detection and not self.at_entrance:
            return
        current_time = self.get_clock().now()
        if (current_time - self.last_processed_time).nanoseconds < 1e9:  # Process at most once per second
            return
        
        if self.at_entrance:
            self.last_processed_time = current_time
            thread = threading.Thread(target=self.detect_colour, args=(data,))
            thread.start()
            self.threads.append(thread)
            
                          
    def detect_colour(self, data):
        try:
            # Convert the received image into a opencv image
            cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8") 
        
            # Set the upper and lower bounds for the green and red
            hsv_green_lower = np.array([45, 50, 50])  # Lower bounds of hue, saturation, value
            hsv_green_upper = np.array([75, 255, 255])  # Upper bounds
           
            hsv_red_lower1 = np.array([0, 120, 70])
            hsv_red_upper1 = np.array([10, 255, 255])
            hsv_red_lower2 = np.array([170, 120, 70])
            hsv_red_upper2 = np.array([180, 255, 255])

        
            # Convert the rgb image into a hsv image
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Filter out everything but a particular colour using the cv2.inRange() method
            green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
            mask_red1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
            mask_red2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
            red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    
            contours_green, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_red, _ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            with self.lock:
                if contours_green:
                    c_g = max(contours_green, key=cv2.contourArea)
                    if cv2.contourArea(c_g) > 2000 and self.is_circle(c_g) and self.found_button == False:
                        self.found_button = True
                        self.detect_colour_green = True
                        self.navigate_to_center(self.current_module)

                elif contours_red:
                    c_r = max(contours_red, key=cv2.contourArea)
                    if cv2.contourArea(c_r) > 2000 and self.is_circle(c_r) and self.found_button == False:
                        self.found_button = True
                        self.detect_colour_red = True
                        opposite_module = 'module_1' if self.current_module == 'module_2' else 'module_2'
                        self.navigate_to_center(opposite_module)
        
        except CvBridgeError as err:
            self.get_logger().error(f'Failed to convert image: {str(err)}')         
    
    # Function to check if contour is a circle
    def is_circle(self, contour, threshold=0.8):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        return circularity > threshold

     # ----------------------------------------------------------------------------------------------------------------------------------- # 
    
             
    # -------------------------------------------------------- WINDOW DETECTION ---------------------------------------------------------- # 
    def detect_window(self):
        try:
            if self.latest_image_data is None:
                self.get_logger().error('No image data available for window detection.')
                return
            
            # Convert the image to an OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(self.latest_image_data, "bgr8")
            window_label = self.window_detection(cv_image)

            # Perform Canny edge detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours in the edge-detected image
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
            if window_label == 'window':
                # Loop over the contours to find quadrilaterals
                for contour in contours:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) == 4:
                        self.get_logger().info("4 Contour")
                        enlargement_factor = 1.8 # Adjust the factor to control zoom-out level
                        window_corners = approx.reshape(4, 2)
                        window_corners = self.sort_corners(window_corners)
                        center = np.mean(window_corners, axis=0)
                        enlarged_corners = (window_corners - center) * enlargement_factor + center
                        
                        # Ensure enlarged_corners is float32
                        enlarged_corners = enlarged_corners.astype(np.float32)
                        # window_corners = approx.reshape(4, 2)
                        # window_corners = np.float32(sorted(window_corners, key=lambda x: (x[1], x[0])))
                        # desired_velocity = Twist()
                        # desired_velocity.linear.x = 0.1  
                        # self.publisher.publish(desired_velocity)
                        if cv2.contourArea(contour) > 30000:
                            self.contour_detection=True
                            self.get_logger().info(f'Contour area: {cv2.contourArea(contour)}, stopping.')
                            planet_label = self.detect_planet(cv_image)
                            self.capture_image(cv_image, planet_label, enlarged_corners)
                            self.contour_detection=False
                        else:
                            self.get_logger().info(f'Contour area: {cv2.contourArea(contour)}, moving forqward.')
                            # self.publisher.publish(desired_velocity)
                            
                return  # Exit after saving one image

        except CvBridgeError as err:
            self.get_logger().error(f'Failed to convert image: {str(err)}')

    
    def window_detection(self, image):
        try:
            result = self.windowmodel.predict(image, imgsz=960, conf=0.5, iou=0.5)
            
            class_name = self.windowmodel.names
            boxes = result[0].boxes.cpu().numpy()
            labels = boxes.cls.astype(int).tolist()
            window_count = sum(1 for label in labels if class_name[label] == "window")

            for box, label in zip(boxes, labels):
                return class_name[label]  # Move this line inside the loop
            
        except CvBridgeError as err:
            self.get_logger().error(f'Failed to convert image: {str(err)}')
            
    def detect_planet(self, image):
        try:
            result = self.planetmodel.predict(image, imgsz=960, conf=0.5, iou=0.5)
            
            class_name = self.planetmodel.names
            boxes = result[0].boxes.cpu().numpy()
            labels = boxes.cls.astype(int).tolist()

            for box, label in zip(boxes, labels):
                return class_name[label]  # Move this line inside the loop
            
        except CvBridgeError as err:
            self.get_logger().error(f'Failed to convert image: {str(err)}')
    
    def create_group_directory(self, group_name='group15'):
        home_directory = os.path.expanduser('~')
        group_directory = os.path.join(home_directory, group_name)

        # Create the directory if it doesn't exist
        if not os.path.exists(group_directory):
            os.makedirs(group_directory)
        return group_directory
    
    def capture_image(self, cv_image, planet_label, window_corners):
        # Get the group directory
        group_directory = self.create_group_directory('group15')

        
        # Define the destination corners for the perspective transformation
        output_size = (1920, 1440)   # Higher resolution for better quality
        dst_corners = np.array([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]], dtype='float32')
        M = cv2.getPerspectiveTransform(window_corners, dst_corners)
        aligned_image = cv2.warpPerspective(cv_image, M, output_size)

        if planet_label:
            if planet_label not in self.detected_planets:
                # Save the original image in the 'group15' directory
                window_filename = f'window{self.window_counter}.png'
                window_path = os.path.join(group_directory, window_filename)
                # cv2.imwrite(window_path, cv_image)
                cv2.imwrite(window_path, aligned_image)
                self.window_counter += 1

                # Save specific images for Earth and Moon in the 'group15' directory
                if planet_label == 'Earth':
                    earth_path = os.path.join(group_directory, 'viewEarth.png')
                    # cv2.imwrite(earth_path, cv_image)
                    cv2.imwrite(earth_path, aligned_image)
                elif planet_label == 'Moon':
                    moon_path = os.path.join(group_directory, 'viewMoon.png')
                    # cv2.imwrite(moon_path, cv_image)
                    cv2.imwrite(moon_path, aligned_image)

                # Add the detected planet label to the set
                self.detected_planets.add(planet_label)
                
    def sort_corners(self, corners):
        rect = np.zeros((4, 2), dtype="float32")

        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]

        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]

        return rect         
            
    # ----------------------------------------------------------------------------------------------------------------------------------- # 
           
    
    # ------------------------------------------------------------ MOVEMENTS ------------------------------------------------------------ #                  
    def stop(self):
        desired_velocity = Twist()  
        self.publisher.publish(desired_velocity)
         
    def rotate_to_detect_button(self):
        def rotation_task():
            desired_velocity = Twist()
            desired_velocity.angular.z = 0.5
            
            start_time = time.time()
            rotation_duration = 10  

            while time.time() - start_time < rotation_duration:
                if self.found_button == True: 
                    self.get_logger().info('Stopping rotation.')
                    break
                self.publisher.publish(desired_velocity)
                self.rate.sleep()
            
            self.stop()

        # Run the rotation in a separate thread
        threading.Thread(target=rotation_task).start()   
    
    def rotate_randomly(self, continue_moving=False):
        desired_velocity = Twist()
        desired_velocity.angular.z = math.pi / 10

        while self.detect_obstacle():
            self.publisher.publish(desired_velocity)
            self.rate.sleep()

        self.stop()

        if continue_moving:
            self.move_forward_after_rotation()    

    def rotate_at_boundary(self):
        desired_velocity = Twist()
        desired_velocity.angular.z = (math.pi/4)

        rotation_duration = math.pi / (2 * abs(desired_velocity.angular.z))

        start_time = time.time()
        while time.time() - start_time < rotation_duration:
            self.get_logger().info('Out-of-boundary rotate')
            self.publisher.publish(desired_velocity)
            self.rate.sleep()

        self.stop()
        self.move_forward_after_rotation()

    def move_forward_after_rotation(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.5
        move_duration = 2  # Move forward for 2 seconds

        start_time = time.time()
        while time.time() - start_time < move_duration:
            self.publisher.publish(desired_velocity)
            self.get_logger().info('Moving forward')
            self.rate.sleep()

        self.stop()   

    def walk_forward_until_obstacle(self):
        # def walking_task():
            while rclpy.ok():
                current_position = self.get_current_position()
                if current_position and current_position[1] < self.y_boundary:
                    self.reached_boundary = True
                    self.get_logger().info('Boundary reached. Rotating.')
                    self.stop()
                    self.rotate_at_boundary()
                elif self.detect_obstacle():
                    self.reached_boundary = False
                    self.get_logger().info('Obstacle detected.')
                    self.stop()
                    self.rotate_randomly(continue_moving=True)    
                else:
                    self.reached_boundary = False
                    desired_velocity = Twist()
                    desired_velocity.linear.x = 0.5
                    self.publisher.publish(desired_velocity)
                    # self.rate.sleep()
                self.rate.sleep() 
                # if self.start_window_detection:
                #     with self.lock:
                #         self.detect_window()    

        # threading.Thread(target=walking_task).start()   
        
    def laser_callback(self, msg):
        self.laser_data = msg
        
        threshold_distance = 1.0 # Distance in meters to consider as an obstacle
        front_distances = msg.ranges[:30] + msg.ranges[-30:]  # Front laser scan ranges
        self.closest_distance = min(front_distances)
        
        if any(distance < threshold_distance for distance in front_distances):
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False  
            
    def detect_obstacle(self):
        return self.obstacle_detected       
            
    # ----------------------------------------------------------------------------------------------------------------------------------- #  
    
    def check_images_present(self, directory='group15'):
        home_directory = os.path.expanduser('~')
        img_directory = os.path.join(home_directory, directory)
        image_earth_path = os.path.join(img_directory, 'viewEarth.png')
        image_moon_path = os.path.join(img_directory, 'viewMoon.png')
        
        return os.path.exists(image_earth_path) and os.path.exists(image_moon_path)
                  
def main():
    def signal_handler(sig, frame):
        # TODO: make sure the robot stops properly here?
        robonaut.stop() 
        rclpy.shutdown()

    rclpy.init(args=None)
    robonaut = RoboNaut()

    signal.signal(signal.SIGINT, signal_handler)
    
    thread = threading.Thread(target=rclpy.spin, args=(robonaut,), daemon=True)
    thread.start()
    
    # robonaut.navigate_to(robonaut.coordinates.module_1.entrance.x, robonaut.coordinates.module_1.entrance.y, 0.0)
    robonaut.navigate_to_nearest_entrance()
    # robonaut.detect_window()
    start_time = time.time()
    images_found = False
    
    try:
        while rclpy.ok() and not images_found:
            if robonaut.check_images_present():
                images_found = True
                break
            if time.time() - start_time > 300:  # 5 minutes
                break
            if robonaut.at_center:
                robonaut.start_window_detection = True
                robonaut.window_detection_event.set()  # Activate the event once
            time.sleep(0.1)
            pass
    except ROSInterruptException:
        pass
    
    finally:
        # Stop the robot and shutdown rclpy
        robonaut.stop()
        rclpy.shutdown()

    if images_found or time.time() - start_time > 300:
        processor = ImageProcessor()
        processor.process_images()
        
        
        

if __name__ == "__main__":
    main()
