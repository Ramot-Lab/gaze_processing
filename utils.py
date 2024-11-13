import cv2
import json

def get_panel_edges(cv2_image) -> list:
    """
    Given an image, allows the user to manually select 4 points on the image,
    presenting the selected dots as red dots on the image and gets approval from the user.
    Returns the 4 points as an array of [(x1, y1), ..., (x4, y4)]
    """
    # Store the selected points
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the selected point to the list
            if len(points) < 4:
                points.append((x, y))
                # Draw a red dot on the image at the selected point
                cv2.circle(cv2_image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Points", cv2_image)

    # Clone the image for refreshing if needed
    image_clone = cv2_image.copy()
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", mouse_callback)

    # Main loop to show the image and collect points
    while True:
        cv2.imshow("Select Points", cv2_image)
        key = cv2.waitKey(1) & 0xFF

        # Reset if user presses 'r'
        if key == ord('r'):
            points = []
            cv2_image = image_clone.copy()
            cv2.imshow("Select Points", cv2_image)

        # Confirm points if user presses 'c' and 4 points are selected
        elif key == ord('c') and len(points) == 4:
            break

        # Exit without saving points if user presses 'q'
        elif key == ord('q'):
            points = []
            break

    cv2.destroyAllWindows()
    return points

def plot_points_on_image(cv2_image, points):
    """
    Given an image and a set of points, plots red dots at each point on the image.
    Displays the image with the points marked.
    
    Parameters:
    - cv2_image: The OpenCV image on which to plot the points.
    - points: A list of (x, y) tuples representing points to plot on the image.
    """
    # Clone the image to avoid modifying the original
    image_with_dots = cv2_image.copy()

    # Plot each point as a red dot
    for (x, y) in points:
        cv2.circle(image_with_dots, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red dot with a radius of 5

    # Display the image
    cv2.imshow("Image with Points", image_with_dots)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def map_panel_into_dot_points(panel_image):
    json_path = "center_locations.json"
    panel = cv2.imread(panel_image)
    width, hight = panel.shape[:2]
    images_in_row = 15
    rows_in_panel = 8
    panel_edges = get_panel_edges(panel)
    dist_between_images_in_row = (panel_edges[1][0]-panel_edges[0][0])/14
    dist_between_rows_in_panel = (panel_edges[2][1] - panel_edges[0][1])/7
    center_locations = []
    for dot_j in range(rows_in_panel):
        for dot_i in range(images_in_row):
            center_locations.append((panel_edges[0][0] + (dist_between_images_in_row*dot_i),
                                        panel_edges[0][1] +  (dist_between_rows_in_panel*dot_j)))
    # plot_points_on_image(panel, center_locations)
    data = {"center_locations":center_locations}
    with open(json_path, "w") as f:
        json.dump(data, f, indent = 4)
    return center_locations


