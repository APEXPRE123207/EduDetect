import cv2
import numpy as np

def mask_image(image_path, rgb_values):

    target_rgb =  rgb_values
    target_rgb_np = np.uint8([[[target_rgb[0], target_rgb[1], target_rgb[2]]]])
    target_hsv = cv2.cvtColor(target_rgb_np, cv2.COLOR_RGB2HSV)
    h_sample, s_sample, v_sample = target_hsv[0][0]
    # print("Target HSV:", target_hsv[0][0])

    h_tolerance = 20  
    s_tolerance = 60  
    v_tolerance = 60  

    lower_bound = np.array([
        max(h_sample - h_tolerance, 0),
        max(s_sample - s_tolerance, 0),
        max(v_sample - v_tolerance, 0)
    ])
    upper_bound = np.array([
        min(h_sample + h_tolerance, 179),
        min(s_sample + s_tolerance, 255),
        min(v_sample + v_tolerance, 255)
    ])

    # print("Lower HSV bound:", lower_bound)
    # print("Upper HSV bound:", upper_bound)


    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Please check the path.")
        exit()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=0)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=0)


    contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)            



    scale_factor = 1.5  
    display_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    display_mask  = cv2.resize(mask_clean, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    mask_clean=cv2.bitwise_not(mask_clean)
    
    masked_image_path = image_path.replace(".jpg", "_masked.jpg")
    cv2.imwrite(masked_image_path, mask_clean)

    print(f"Masked image saved at: {masked_image_path}")
    return masked_image_path

