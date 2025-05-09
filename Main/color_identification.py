import cv2
import pandas as pd
import random
from collections import Counter
def color_identify(image_path, color_csv_path):
    clicked = False
    r = g = b = xpos = ypos = 0
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not read the image.")
        exit()

    # Load colors CSV file
    index = ["color", "color_name", "hex", "R", "G", "B"]
    csv = pd.read_csv(r'..\Resources\colors4.csv', names=index, header=None)

    def getColorName(R, G, B):
        """Returns the closest color name from the CSV based on minimum RGB difference."""
        minimum = float('inf')
        cname = ""
        for i in range(len(csv)):
            d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
            if d < minimum:
                minimum = d
                cname = csv.loc[i, "color_name"]
        return cname

    # Get the center color of the image (OpenCV returns BGR values)
    height, width, _ = img.shape
    center_x = width // 2
    center_y = height // 2
    b, g, r = img[center_y, center_x]
    b, g, r = int(b), int(g), int(r)
    
    center_rgb = (r, g, b)
    center_color_name = getColorName(r, g, b)
    text = f"{center_color_name}"

    # Mark the center and display the color name on the image
    cv2.circle(img, (center_x, center_y), 5, (255, 255, 255), -1)
    cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)
    text_color = (0, 0, 0) if (r + g + b) >= 600 else (255, 255, 255)
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

    region_size = 100

    start_x = max(center_x - region_size // 2, 0)
    end_x = min(center_x + region_size // 2, width)
    start_y = max(center_y - region_size // 2, 0)
    end_y = min(center_y + region_size // 2, height)

    num_samples = 50
    sampled_colors = []

    for _ in range(num_samples):
        x = random.randint(start_x, end_x - 1)
        y = random.randint(start_y, end_y - 1)
        # Note: OpenCV uses BGR format for colors
        color = tuple(img[y, x])
        sampled_colors.append(color)


    color_counts = Counter(sampled_colors)
    most_common_color, count = color_counts.most_common(1)[0]  # most_common_color is a tuple (B, G, R)

    # Convert most common color to RGB order for clarity
    mc_b, mc_g, mc_r = most_common_color
    most_common_rgb = (mc_r, mc_g, mc_b)
    most_common_color_name = getColorName(mc_r, mc_g, mc_b)

    # Compare the center pixel's color and the most common color from the sampled region
    # if center_color_name == most_common_color_name:
    #     print(f"The center color and the most common color are the same: {center_color_name} (RGB: {center_rgb})")
    # else:
    #     print(f"The center color is {center_color_name} (RGB: {center_rgb}) but the most common color in the region is {most_common_color_name} (RGB: {center_rgb})")

    return center_rgb,center_color_name
