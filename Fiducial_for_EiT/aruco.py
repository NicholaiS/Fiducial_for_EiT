import cv2
import numpy as np
from icecream import ic

def main():
    cap = cv2.VideoCapture(0)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        ic(markerCorners)

        frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)

        if markerIds is not None:
            for corners in markerCorners:
                corners = corners[0] 
                center_x = int(np.mean(corners[:, 0]))
                center_y = int(np.mean(corners[:, 1]))

                cv2.circle(frame_markers, (center_x, center_y), 5, (0, 255, 0), -1)

                ic(f"Marker center: ({center_x}, {center_y})")

        frame_resize = cv2.resize(frame_markers, (1080, 720))
        cv2.imshow("frame", frame_resize)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        if k == ord('p'):
            cv2.waitKey(100000)

    cap.release()
    cv2.destroyAllWindows()

main()
