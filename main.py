import cv2 as cv
import time

if __name__ == "__main__":
    capture = cv.VideoCapture(cv.samples.findFileOrKeep("clip_v0.0.0.mp4"))  
    if not capture.isOpened:
        print("Open failure.")
    
    cv.waitKey(10000)
    color = (0, 255, 128)
    size = 2
    while True:
        ret,frame = capture.read()
        if ret is False:
            print("\nVideo ended")
            break

        target = (frame > 100).nonzero()
        target_x = target[1].mean()
        target_y = target[0].mean()
        width  = target[1].ptp()
        height = target[0].ptp()

        bounds_first = (int(target_x-width/2), int(target_y-height/2))
        bounds_second = (int(target_x+width/2),int(target_y+height/2))
        cv.rectangle(frame, bounds_first, bounds_second, color, size)
        print(f"\r X: {target_x:7.3f} Y: {target_y:7.3f} W: {width:4} H: {height:4}", end="")

        cv.imshow("Tracker v0.0.0", frame)
        if cv.waitKey(1) == ord('q'):
            break
