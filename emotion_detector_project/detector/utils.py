import cv2

def draw_label(frame, text, x, y):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2)
