
import cv2
import numpy as np
#
# class ContourHandler:
#     def __init__(self, image, contours):
#         """Initialize with an image and detect contours."""
#         self.image = image
#         self.display_img = self.image.copy()
#         self.contours = contours
#         self.points = [None, None]
#         self.dragging_point = None
#         self.selected_contour = None  # Track the selected contour
#
#     def get_largest_contour(self):
#         """Return the largest contour based on area."""
#         return max(self.contours, key=cv2.contourArea) if self.contours else None
#
#     def find_nearest_contour_point(self, point):
#         """Find the nearest point on the selected contour."""
#         if self.selected_contour is None:
#             return None
#
#         dists = np.linalg.norm(self.selected_contour.reshape(-1, 2) - point, axis=1)
#         nearest_idx = np.argmin(dists)
#         return tuple(self.selected_contour[nearest_idx][0])
#
#     def _get_clicked_contour_index(self, click_pos):
#         """Check if the click is near any contour."""
#         for i, contour in enumerate(self.contours):
#             dists = np.linalg.norm(contour.reshape(-1, 2) - click_pos, axis=1)
#             if np.min(dists) < 10:  # Within 10 pixels
#                 return i  # Return the index of the selected contour
#         return None  # No contour found
#
#     def _get_clicked_point_index(self, click_pos):
#         """Check if the click is near any of the selected points."""
#         for i, point in enumerate(self.points):
#             if point is not None and np.linalg.norm(np.array(point) - click_pos) < 10:
#                 return i
#         return None  # No nearby point found
#
#     def _mouse_callback(self, event, x, y, flags, param):
#         """Handle mouse events for placing and dragging points."""
#         click_pos = np.array([x, y])
#
#         if event == cv2.EVENT_LBUTTONDOWN:
#             # Check if the click is near any existing point
#             point_index = self._get_clicked_point_index(click_pos)
#             if point_index is not None:
#                 # Start dragging the clicked point
#                 self.dragging_point = point_index
#             else:
#                 # Check if the click is near any contour to select it
#                 contour_index = self._get_clicked_contour_index(click_pos)
#                 if contour_index is not None:
#                     self.selected_contour = self.contours[contour_index]  # Select the clicked contour
#                     # Try to place points on the selected contour
#                     for i in range(2):
#                         if self.points[i] is None:
#                             self.points[i] = self.find_nearest_contour_point(click_pos)
#                             self.dragging_point = i  # Start dragging the new point
#                             break
#
#         elif event == cv2.EVENT_MOUSEMOVE and self.dragging_point is not None:
#             # Move the selected point along the selected contour
#             self.points[self.dragging_point] = self.find_nearest_contour_point(click_pos)
#
#         elif event == cv2.EVENT_LBUTTONUP:
#             # Stop dragging
#             self.dragging_point = None
#
#     def set_mouse_callback(self, window_name="Image"):
#         """Set the OpenCV mouse callback for interaction."""
#         cv2.setMouseCallback(window_name, self._mouse_callback)
#
#     def update_display(self):
#         """Redraw the image with contours and the selected points."""
#         self.display_img = self.image.copy()  # Reset the display image
#
#         # Draw all contours
#         for contour in self.contours:
#             cv2.drawContours(self.display_img, [contour], -1, (0, 255, 0), 2)
#
#         # Highlight the selected contour
#         if self.selected_contour is not None:
#             cv2.drawContours(self.display_img, [self.selected_contour], -1, (255, 0, 0), 2)
#
#         # Draw the two selected points, if available
#         for point in self.points:
#             if point is not None:
#                 cv2.circle(self.display_img, point, 5, (0, 0, 255), -1)
#
#     def display_image(self, window_name="Image"):
#         """Show the updated image."""
#         self.update_display()
#         cv2.imshow(window_name, self.display_img)
#
#     def wait_for_key(self, window_name="Image"):
#         """Keep displaying the image until 'q' is pressed."""
#         while True:
#             self.display_image(window_name)
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#         cv2.destroyAllWindows()
#
#     def get_selected_points(self):
#         """Return the two selected points."""
#         return self.points


class ContourHandler:
    def __init__(self, image, contours):
        """Initialize with an image and detect contours."""
        self.image = image
        self.display_img = self.image.copy()
        self.contours = contours
        self.points = [None, None]
        self.dragging_point = None
        self.selected_contour = None  # Track the selected contour
        self.contour_only_mode = True  # Mode to allow contour-only selection

    def get_largest_contour(self):
        """Return the largest contour based on area."""
        return max(self.contours, key=cv2.contourArea) if self.contours else None

    def find_nearest_contour_point(self, point):
        """Find the nearest point on the selected contour."""
        if self.selected_contour is None:
            return None

        dists = np.linalg.norm(self.selected_contour.reshape(-1, 2) - point, axis=1)
        nearest_idx = np.argmin(dists)
        return tuple(self.selected_contour[nearest_idx][0])

    def _get_clicked_contour_index(self, click_pos):
        """Check if the click is near any contour."""
        for i, contour in enumerate(self.contours):
            dists = np.linalg.norm(contour.reshape(-1, 2) - click_pos, axis=1)
            if np.min(dists) < 10:  # Within 10 pixels
                return i  # Return the index of the selected contour
        return None  # No contour found

    def _get_clicked_point_index(self, click_pos):
        """Check if the click is near any of the selected points."""
        for i, point in enumerate(self.points):
            if point is not None and np.linalg.norm(np.array(point) - click_pos) < 10:
                return i
        return None  # No nearby point found

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for placing and dragging points."""
        click_pos = np.array([x, y])

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.contour_only_mode:
                # Select a contour without placing points
                contour_index = self._get_clicked_contour_index(click_pos)
                if contour_index is not None:
                    self.selected_contour = self.contours[contour_index]
            else:
                # Check if the click is near any existing point
                point_index = self._get_clicked_point_index(click_pos)
                if point_index is not None:
                    # Start dragging the clicked point
                    self.dragging_point = point_index
                else:
                    # Check if the click is near any contour to select it
                    contour_index = self._get_clicked_contour_index(click_pos)
                    if contour_index is not None:
                        self.selected_contour = self.contours[contour_index]  # Select the clicked contour
                        # Try to place points on the selected contour
                        for i in range(2):
                            if self.points[i] is None:
                                self.points[i] = self.find_nearest_contour_point(click_pos)
                                self.dragging_point = i  # Start dragging the new point
                                break

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_point is not None:
            # Move the selected point along the selected contour
            self.points[self.dragging_point] = self.find_nearest_contour_point(click_pos)

        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            self.dragging_point = None

    def set_mouse_callback(self, window_name="Image"):
        """Set the OpenCV mouse callback for interaction."""
        cv2.setMouseCallback(window_name, self._mouse_callback)

    def update_display(self):
        """Redraw the image with contours and the selected points."""
        self.display_img = self.image.copy()  # Reset the display image

        # Draw all contours
        for contour in self.contours:
            if cv2.arcLength(contour, True) > 20:
                cv2.drawContours(self.display_img, [contour], -1, (0, 255, 0), 2)

        # Highlight the selected contour
        if self.selected_contour is not None:
            cv2.drawContours(self.display_img, [self.selected_contour], -1, (255, 0, 0), 2)

        # Draw the two selected points, if available
        for point in self.points:
            if point is not None:
                cv2.circle(self.display_img, point, 5, (0, 0, 255), -1)

    def display_image(self, window_name="Image"):
        """Show the updated image."""
        self.update_display()
        cv2.imshow(window_name, self.display_img)

    def wait_for_key(self, window_name="Image"):
        """Keep displaying the image until 'q' is pressed."""
        while True:
            self.display_image(window_name)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                # Toggle contour-only mode
                self.contour_only_mode = not self.contour_only_mode
                print("Contour-only mode:", self.contour_only_mode)

        cv2.destroyAllWindows()

    def get_selected_points(self):
        """Return the two selected points."""
        return self.points
