import cv2 as cv
import numpy as np
import time
from sys import exit
from datetime import datetime

# Colors
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# IA
net = cv.dnn.readNetFromCaffe("model/MobileNetSSD_deploy.prototxt", "model/MobileNetSSD_deploy.caffemodel")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# dictionary with the object class id and names on which the model is trained
classNames = { 
    # 0: 'background',
    # 1: 'aeroplane',
    # 2: 'bicycle', 
    # 3: 'bird', 
    # 4: 'boat',
    # 5: 'bottle', 
    # 6: 'bus', 
    # 7: 'car', 
    # 8: 'cat', 
    # 9: 'chair',
    # 10: 'cow', 
    # 11: 'diningtable', 
    # 12: 'dog',
    # 13: 'horse',
    # 14: 'motorbike', 
    15: 'person',
    # 16: 'pottedplant',
    # 17: 'sheep',
    # 18: 'sofa', 
    # 19: 'train', 
    # 20: 'tvmonitor'
}

# Camera
class Camera:
    def __init__(self, id: int | str = 0):
        self._camera = cv.VideoCapture(id)
        self._last_frame = None
        if not self._camera.isOpened():
            exit("Camera couldn't be initialized")
    
    def get_frame(self) -> cv.Mat:
        return self._camera.read()
    
    def FPS(self) -> int:
        return self._camera.get(cv.CAP_PROP_FPS)
    
    def release(self) -> None:
        self._camera.release()

def main() -> None:
    camera = Camera()
    motion_detected_icon = cv.imread("assets/motion_detected.png")
    if motion_detected_icon is None:
        print("Program couldn't load assets")
        return
    
    FPS_CAP = 1/camera.FPS()
    motion_detected_icon = cv.resize(motion_detected_icon, (16, 16))
    motion_detected_icon = cv.cvtColor(motion_detected_icon, cv.COLOR_BGR2BGRA)
    last_frame = None

    while True:
        frame_start = time.time()

        ret, camera_frame = camera.get_frame()
        if not ret:
            print("Failed to get frame")
            continue

        display_frame = camera_frame

        if last_frame is not None:
            motion = detect_motion(last_frame, camera_frame) 
            # print(f"Difference [0, 1] {motion}")
            if motion > 0.0175:
                display_frame = overlay_img(cv.cvtColor(camera_frame, cv.COLOR_BGR2BGRA), motion_detected_icon, (0, 26))

        display_frame = detect_human(cv.cvtColor(display_frame, cv.COLOR_BGRA2BGR))
        
        frame_brightness = get_brightness(camera_frame)
        # print(f"Brightness [0, 1]: {frame_brightness}")
        if frame_brightness < 0.325:
            display_frame = apply_nightvision(display_frame)
        
        display_frame = display_time(display_frame, (0, 18))

        last_frame = camera_frame # Update last_frame with the current frame for the next iteration

        # FPS CAP
        delta_time = time.time() - frame_start
        if delta_time < FPS_CAP:
            time.sleep(FPS_CAP - delta_time) # stabilize fps 
            delta_time = FPS_CAP
        
        # display_frame = outlined_text(display_frame, f"FPS: {1//delta_time}", (512, 18), 0.8)
        cv.imshow("Scam", display_frame)

        key = cv.waitKey(1)
        if key == 27: # Press 'Esc' to exit the loop
            break
    
    camera.release()
    cv.destroyAllWindows()

def overlay_img(background: cv.Mat, foreground: cv.Mat, pos: tuple) -> cv.Mat:
    x, y = pos
    h, w = foreground.shape[:2]

    x_end = min(x + w, background.shape[1])
    y_end = min(y + h, background.shape[0])
    if x_end <= 0 or y_end <= 0 or x >= background.shape[1] or y >= background.shape[0]:
        return background 

    foreground = foreground[:y_end - y, :x_end - x]

    roi = background[y:y_end, x:x_end]

    # verify if it has an aplha channel
    if foreground.shape[2] == 4:  # BGRA
        alpha_foreground = foreground[:, :, 3]/255.0
        alpha_background = 1.0 - alpha_foreground

        for c in range(3):  # B, G, R
            roi[:, :, c] = alpha_foreground*foreground[:, :, c] + alpha_background*roi[:, :, c]

        roi[:, :, 3] = (alpha_foreground*255 + alpha_background*roi[:, :, 3])

    else:
        roi[:, :, :3] = foreground[:, :, :3]

    # Atualiza a região no fundo
    background[y:y_end, x:x_end] = roi
    return background

def outlined_text(img: cv.Mat, text: str, pos: tuple, scale: float = 1, fg_color: tuple = (255, 255, 255), outline: tuple = (0, 0, 0), thickness: int = 2) -> cv.Mat:
    cv.putText(img, text, pos, cv.FONT_HERSHEY_DUPLEX, scale, outline, thickness*2)
    cv.putText(img, text, pos, cv.FONT_HERSHEY_DUPLEX, scale, fg_color, thickness)

    return img

def get_brightness(img: cv.Mat) -> float:
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Normalize to range [0, 1]
    saturation = np.mean(img_hsv[:, :, 1])/255.0
    brightness = np.mean(img_hsv[:, :, 2])/255.0

    return 0.7*brightness + 0.3*(1 - saturation)

def apply_nightvision(img: cv.Mat, gamma: float = 2) -> cv.Mat:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)

    # Normalize to range [0, 1] the scale back
    img = np.power(img/255.0, gamma)*255.0

    return np.clip(img, 0, 255).astype(np.uint8)

def detect_human(frame: cv.Mat, confidence_thereshold: float=0.5) -> cv.Mat:
    w, h = frame.shape[:2] 
    # construct a blob from the image
    blob = cv.dnn.blobFromImage(frame, scalefactor=1/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
    
    # blob object is passed as input to the object
    net.setInput(blob)

    # network prediction
    detections = net.forward() 

    # detections array is in the format 1, 1, N, 7, where N is the detected bounding boxes
    # for each detection, the description (7) contains : [image_id, label, conf, x_min, y_min, x_max, y_max]
    person = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # get class id
        class_id = int(detections[0, 0, i, 1])
        # set confidence level threshold to filter weak predictionsS
        if confidence > confidence_thereshold and class_id == 15:
            person += 1

            # scale to the frame
            x_top_left = int(detections[0, 0, i, 3]*w) 
            y_top_left = int(detections[0, 0, i, 4]*h)
            x_bottom_right = int(detections[0, 0, i, 5]*w)
            y_bottom_right = int(detections[0, 0, i, 6]*h)
            
            # draw bounding box around the detected object
            cv.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), GREEN)
            
            text = f"{(confidence*100):.2f}% Person: {person}"
            (text_w, text_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)

            # Define the rectangle's top-left and bottom-right corners
            rect_top_left = (x_top_left, y_top_left - text_h - baseline)
            rect_bottom_right = (x_top_left + text_w, y_top_left)

            cv.rectangle(frame, rect_top_left, rect_bottom_right, BLACK, cv.FILLED)
            cv.putText(frame, text, (x_top_left, y_top_left), cv.FONT_HERSHEY_DUPLEX, 0.5, RED)
    
    frame = outlined_text(frame, f"Total people: {person}", (20, 38), 0.5, WHITE)
    return frame

def detect_motion(last_frame: cv.Mat, current_frame: cv.Mat) -> float:
    # Convert to grayscale
    if get_brightness(current_frame) < 0.375:
        last_frame = apply_nightvision(last_frame)
        current_frame = apply_nightvision(current_frame)
    else:
        last_frame = cv.cvtColor(last_frame, cv.COLOR_BGR2GRAY)
        current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    
    # apply blur
    last_frame = cv.GaussianBlur(last_frame, (5, 5), 0)
    current_frame = cv.GaussianBlur(current_frame, (5, 5), 0)

    # Compute absolute difference and Normalize to range [0, 1]
    return np.mean(cv.absdiff(last_frame, current_frame))/255.0

def display_time(frame: cv.Mat, pos: tuple) -> cv.Mat:
    return outlined_text(frame, datetime.now().strftime("%m/%d/%Y-%H:%M:%S"), pos, 0.8)

if __name__ == "__main__":
    main()