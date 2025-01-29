# import cv2
# import numpy as np
#
# # Global variables for saved points
# saved_points = []
#
# def find_contours(image):
#     """Detect contours in the image."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours
#
# def parameterize_contour(contour):
#     """Compute the cumulative arc length of each point along the contour."""
#     arc_lengths = [cv2.arcLength(contour[:i+1], False) for i in range(len(contour))]
#     total_length = arc_lengths[-1]
#     normalized_arc_lengths = [l / total_length for l in arc_lengths]  # Normalize to [0, 1]
#     return normalized_arc_lengths
#
# def find_matching_points(contour, saved_rel_positions):
#     """Find the points on the contour that correspond to the saved relative positions."""
#     arc_lengths = parameterize_contour(contour)
#     matching_points = []
#
#     for rel_pos in saved_rel_positions:
#         # Find the index where the arc length is closest to the saved relative position
#         idx = np.argmin(np.abs(np.array(arc_lengths) - rel_pos))
#         matching_points.append(tuple(contour[idx][0]))
#
#     return matching_points
#
# def mouse_callback(event, x, y, flags, param):
#     """Save points on the contour by clicking."""
#     global saved_points
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Find the nearest contour point to the click
#         nearest_point = find_nearest_contour_point(np.array([x, y]), contours[0])
#         print(f"Saved point: {nearest_point}")
#         saved_points.append(nearest_point)
#
# def find_nearest_contour_point(point, contour):
#     """Find the nearest point on a contour to a given point."""
#     dists = np.linalg.norm(contour.reshape(-1, 2) - point, axis=1)
#     nearest_idx = np.argmin(dists)
#     return tuple(contour[nearest_idx][0])
#
# def main():
#     global contours
#
#     # Load and process the original image
#     img = cv2.imread('img.png')
#     contours = find_contours(img)
#
#     # Select the largest contour (for example)
#     contour = max(contours, key=cv2.contourArea)
#
#     # Create a window and set mouse callback to save points
#     cv2.namedWindow('Image')
#     cv2.setMouseCallback('Image', mouse_callback)
#
#     print("Click on two points on the contour to save them.")
#
#     # Display the image and let the user select points
#     while True:
#         display_img = img.copy()
#         cv2.drawContours(display_img, [contour], -1, (0, 255, 0), 2)
#
#         for point in saved_points:
#             cv2.circle(display_img, point, 5, (0, 0, 255), -1)
#
#         cv2.imshow('Image', display_img)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#
#     # Save points as relative positions along the contour
#     rel_positions = parameterize_contour(contour)
#     saved_rel_positions = [
#         rel_positions[np.argmin(np.linalg.norm(contour - np.array(pt), axis=1))]
#         for pt in saved_points
#     ]
#
#     # Load a rotated or transformed version of the image
#     img_transformed = cv2.imread('imgrotate.png')
#     contours_transformed = find_contours(img_transformed)
#     contour_transformed = max(contours_transformed, key=cv2.contourArea)
#
#     # Find matching points on the transformed contour
#     matching_points = find_matching_points(contour_transformed, saved_rel_positions)
#
#     # Display the transformed image with matching points
#     while True:
#         display_img = img_transformed.copy()
#         cv2.drawContours(display_img, [contour_transformed], -1, (0, 255, 0), 2)
#
#         for point in matching_points:
#             cv2.circle(display_img, point, 5, (255, 0, 0), -1)
#
#         cv2.imshow('Transformed Image', display_img)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
# from skimage import io
# from skimage.filters import threshold_otsu
# import matplotlib.pyplot as plt
#
# image = io.imread('medianblur.png', as_gray=True)
# thresh = threshold_otsu(image)
# binary = image > thresh
#
# fig, ax = plt.subplots(ncols=3, figsize=(11, 2.5))
#
# # ax[1].imshow(image, cmap=plt.cm.gray)
# # ax[1].set_title('Original')
# # ax[1].axis('off')
#
# ax[0].hist(image.ravel(), bins=256)
# ax[0].set_title("Otsu's Method Histogram")
# ax[0].axvline(thresh, color='r')
# #
# # ax[2].imshow(binary, cmap=plt.cm.gray)
# # ax[2].set_title('Thresholded')
# # ax[2].set_axis_off()
#
# plt.show()

import pyrealsense2 as rs
import numpy as np
import cv2


class ARC:
    def __init__(self):
        self.pipeline = rs.pipeline()

        # Configure the pipeline to use live feed
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable depth stream
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable color stream

        self.pipeline.start(config)

    def video(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Color Stream", self.on_mouse_click)

        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue  # Skip if frames are not ready

            self.depth_frame = depth_frame

            color_image = np.asanyarray(color_frame.get_data())
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            combined_image = np.hstack((color_image, depth_color_image))
            cv2.imshow("Color Stream", combined_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            depth = self.depth_frame.get_distance(x, y)
            print(f"Depth at ({x}, {y}): {depth:.4f} meters")


if __name__ == '__main__':
    ARC().video()
