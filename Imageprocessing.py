import cv2 
import numpy as np
from matplotlib import pyplot as plt


class RoadSegmentationAPI:
    def __init__(self, bfs_threshold=50, sobel_threshold=100, lbp_threshold=0.5, stride=1, sobel_ksize=3):
        self.params = {
            "bfs_threshold": bfs_threshold,      # BFS threshold
            "sobel_threshold": sobel_threshold,  # Sobel edge threshold
            "lbp_threshold": lbp_threshold,      # LBP threshold
            "stride": stride,                    # Stride size
            "sobel_ksize": sobel_ksize           # Sobel kernel size
        }

    def preprocess(self, image):
        """Convert to grayscale and perform Sobel edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.params["sobel_ksize"])
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.params["sobel_ksize"])
        edges = cv2.magnitude(sobel_x, sobel_y)
        norm_edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray, norm_edges

    def lbp_analysis(self, gray):
        """Perform 3x3 LBP and calculate histogram."""
        h, w = gray.shape
        lbp_image = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i - 1, j - 1] > center) << 7
                code |= (gray[i - 1, j] > center) << 6
                code |= (gray[i - 1, j + 1] > center) << 5
                code |= (gray[i, j + 1] > center) << 4
                code |= (gray[i + 1, j + 1] > center) << 3
                code |= (gray[i + 1, j] > center) << 2
                code |= (gray[i + 1, j - 1] > center) << 1
                code |= (gray[i, j - 1] > center) << 0
                lbp_image[i, j] = code

        hist = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()

        return lbp_image, hist_norm

    def path_search(self, gray, seed_point):
        """BFS-based region growing for road area."""
        h, w = gray.shape
        visited = np.zeros_like(gray, dtype=np.uint8)
        queue = [seed_point]
        road_area = np.zeros_like(gray, dtype=np.uint8)

        while queue:
            x, y = queue.pop(0)
            if visited[x, y] == 0 and gray[x, y] > self.params["bfs_threshold"]:
                visited[x, y] = 1
                road_area[x, y] = 255
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and visited[nx, ny] == 0:
                        queue.append((nx, ny))
        return road_area

    def refine_road_mask(self, image, bfs_mask, edges):
        """Refine road mask with Sobel edges and HSV filtering."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV range for road detection
        lower_bound = np.array([0, 0, 40])  # Dark gray
        upper_bound = np.array([180, 40, 130])  # Light gray

        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Sobel edge mask (binary threshold)
        _, sobel_mask = cv2.threshold(edges, self.params["sobel_threshold"], 255, cv2.THRESH_BINARY)

        # Combine BFS mask, HSV mask, and Sobel edge mask
        combined_mask = cv2.bitwise_and(color_mask, bfs_mask)
        combined_mask = cv2.bitwise_or(combined_mask, sobel_mask)

        # Keep only the lower half of the image
        h, w = combined_mask.shape
        combined_mask[:h // 2, :] = 0

        # Morphological operations: closing -> opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # Larger structure element
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        return cleaned_mask

    def coloring_with_red(self, image, road_mask):
        """Color the road area red with additional conditions."""
        colored_image = image.copy()

        # Only color sufficiently large connected regions
        num_labels, labels = cv2.connectedComponents(road_mask.astype(np.uint8))
        for label in range(1, num_labels):  # Skip background (0)
            current_mask = labels == label
            if np.sum(current_mask) > 500:  # Only process areas larger than 500 pixels
                colored_image[current_mask] = [0, 0, 255]  # Color red

        return colored_image


# Main function
if __name__ == "__main__":
    # Update the image path
    image_path = r"C:\Users\willi\Desktop\imagere\image.jpg"

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
    else:
        api = RoadSegmentationAPI(bfs_threshold=60, sobel_threshold=50, lbp_threshold=0.5, stride=1, sobel_ksize=3)

        # 1. Preprocessing
        gray, edges = api.preprocess(image)

        # 2. Texture Analysis
        lbp_image, lbp_hist = api.lbp_analysis(gray)

        # 3. Path Search (BFS)
        seed_point = (gray.shape[0] - 50, gray.shape[1] // 2)  # Near the center of the road
        bfs_mask = api.path_search(gray, seed_point)

        # 4. Refine Road Mask with Sobel edges
        refined_mask = api.refine_road_mask(image, bfs_mask, edges)

        # 5. Color the Road
        colored_road = api.coloring_with_red(image, refined_mask)

        # Display all results in one figure
        plt.figure(figsize=(20, 15))

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(gray, cmap="gray")
        plt.title("Grayscale Image")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(edges, cmap="gray")
        plt.title("Sobel Edge Detection")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(lbp_image, cmap="gray")
        plt.title("LBP Texture Analysis")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.plot(lbp_hist, color='blue')
        plt.title("LBP Histogram")
        plt.xlabel("Bins")
        plt.ylabel("Frequency")

        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(colored_road, cv2.COLOR_BGR2RGB))
        plt.title("Final Refined Colored Road")
        plt.axis("off")

        plt.tight_layout()
        plt.show()  