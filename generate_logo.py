import numpy as np
import cv2

# Generate clean high-resolution ISRO(logo).png based on the uploaded image
size = 256
img = np.zeros((size, size, 4), dtype=np.uint8)

# Center of logo
cx, cy = 128, 128
r_outer = 78
r_inner = 70
shift_x = 18
shift_y = 6

# Create mask for crescent
mask_outer = np.zeros((size, size), dtype=np.uint8)
mask_inner = np.zeros((size, size), dtype=np.uint8)

cv2.circle(mask_outer, (cx, cy), r_outer, 255, -1, cv2.LINE_AA)
cv2.circle(mask_inner, (cx + shift_x, cy - shift_y), r_inner, 255, -1, cv2.LINE_AA)

crescent = cv2.subtract(mask_outer, mask_inner)

# 4-point star in top-right of crescent opening
star_mask = np.zeros((size, size), dtype=np.uint8)
star_cx, star_cy = cx + 32, cy - 36
star_r_long = 24
star_r_short = 6

pts = []
for i in range(8):
    angle = i * np.pi / 4 - np.pi / 2
    r = star_r_long if i % 2 == 0 else star_r_short
    px = star_cx + r * np.cos(angle)
    py = star_cy + r * np.sin(angle)
    pts.append([px, py])

pts = np.array(pts, dtype=np.int32)
cv2.fillPoly(star_mask, [pts], 255, cv2.LINE_AA)

combined = cv2.bitwise_or(crescent, star_mask)
combined = cv2.GaussianBlur(combined, (3, 3), 0)

# Set white color with alpha
img[:, :, 0] = 255
img[:, :, 1] = 255
img[:, :, 2] = 255
img[:, :, 3] = combined

cv2.imwrite("frontend/public/ISRO(logo).png", img)
print("Saved frontend/public/ISRO(logo).png")
