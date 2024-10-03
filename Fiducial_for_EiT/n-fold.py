from MarkerTracker import MarkerTracker
import cv2
import numpy as np

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Open and move windows
    cv2.namedWindow("Frame")
    cv2.namedWindow("Magnitude image")
    cv2.moveWindow("Magnitude image", 900, 0)

    # Initialize tracker
    tracker = MarkerTracker(
            order=2, 
            kernel_size=25, 
            scale_factor=0.1)
    tracker.track_marker_with_missing_black_leg = False

    # Analyse each frame individually
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        # Scale down the image and convert it to grayscale
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Locate the marker
        pose = tracker.locate_marker(grayscale_image)

        # Extract the marker response from the tracker object
        magnitude = np.sqrt(tracker.frame_sum_squared)

        # Visualise the location of the located marker and indicate the quality
        # of the detected marker by altering the line color.
        # Yellow: bad marker quality
        # Red: high marker quality
        color = (0, 255 - int(255 * pose.quality), 255)
        cv2.line(frame, (0, 0), (pose.x, pose.y), color, 2)

        # Show annotated input image and the magnitude response image.
        cv2.imshow("Magnitude image", magnitude / 315)
        cv2.imshow("Frame", frame)

        # Make it possible to stop the program by pressing 'q'.
        k = cv2.waitKey(30)
        if k == ord('q'):
            break


main()

