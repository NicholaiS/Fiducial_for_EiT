import cv2
import numpy as np
from MarkerTracker import MarkerTracker

# For the Secchi marker the location of the marker and indication of the quality
# of the detected marker is visuallized by altering the line color.
# Yellow: bad marker quality
# Red: high marker quality

def main():
    cap = cv2.VideoCapture(0)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    cv2.namedWindow("Frame: N-fold")
    cv2.namedWindow("Frame: Aruco")
    cv2.moveWindow("Frame: Aruco", 900, 0)

    tracker = MarkerTracker(
            order=2, 
            kernel_size=25, 
            scale_factor=0.1)
    tracker.track_marker_with_missing_black_leg = False

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        
        # Aruco marker
        aruco_found = False
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)

        if markerIds is not None:
            aruco_found = True
            for corners in markerCorners:
                corners = corners[0] 
                center_x = int(np.mean(corners[:, 0]))
                center_y = int(np.mean(corners[:, 1]))

                cv2.circle(frame_markers, (center_x, center_y), 5, (0, 255, 0), -1)

                print(f"Aruco marker center: ({center_x}, {center_y})")

        # N-fold marker
        if not aruco_found:
            grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pose = tracker.locate_marker(grayscale_image)

            color = (0, 255 - int(255 * pose.quality), 255)
            cv2.line(frame, (0, 0), (pose.x, pose.y), color, 2)
            print(f"Secchi marker center: ({pose.x}, {pose.y})")

        cv2.imshow("Frame: N-fold", frame)
        cv2.imshow("Frame: Aruco", frame_markers)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        if k == ord('p'):
            cv2.waitKey(100000)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
