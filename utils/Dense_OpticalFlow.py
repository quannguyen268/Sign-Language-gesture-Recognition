import cv2
import numpy as np
from utils.Data_manipulation import _calc_optical_flow
# Get a VideoCapture object from video and store it in vс
vc = cv2.VideoCapture('/home/quan/PycharmProjects/sign-language-gesture-recognition/all/055_002_004.mp4')
# Read first frame
_, first_frame = vc.read()
# Scale and resize image
resize_dim = 600
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
# Convert to gray scale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Create mask
mask = np.zeros_like(first_frame)
# Set image saturation to maximum value as we do not need it
mask[..., 1] = 255

while (vc.isOpened()):
    # Read a frame from video
    _, frame = vc.read()

    # Convert new frame format`s to gray scale and resize gray frame obtained
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)
#
    # Calculate dense optical flow by Farneback method
    flow = _calc_optical_flow(prev_gray, gray)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # Resize frame size to match dimensions
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    # Open a new window and displays the output frame
    dense_flow = cv2.addWeighted(frame, 1, rgb, 2, 0)
    dense_flow = cv2.cvtColor(dense_flow, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Denseopticalflow', dense_flow)
    # Update previous frame
    prev_gray = gray
#     # Frame are read by intervals of 10 millisecond. The programs breaks out of the while loop when the user presses the ‘q’ key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
#
vc.release()
cv2.destroyAllWindows()