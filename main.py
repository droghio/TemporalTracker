import cv2 as cv
import numpy
import time

if __name__ == "__main__":
    capture = cv.VideoCapture(cv.samples.findFileOrKeep("media/Football/football.mp4"))
    if not capture.isOpened:
        print("Open failure.")
    
    color = (0, 255, 0)
    size = 2
    ddepth = -1
    thres = None
    start_pos = [310,102,39,50]
    start = dict(zip(["x", "y", "w", "h"], start_pos))
    first_frame = True
    kernel = None
    while True:
        ret,frame = capture.read()
        if ret is False:
            print("\nVideo ended")
            break

        if first_frame:
            first_frame = False
            chip = frame[start["y"]:start["y"]+start["h"], start["x"]:start["x"]+start["w"]]
            kernel = chip.astype(float)/(chip.shape[0] * chip.shape[1] * chip.max())
            while True:
                cv.imshow("Tracker v0.0.1", chip)
                if cv.waitKey(100) == ord('q'):
                    break
            continue

        dst = cv.filter2D(frame, ddepth, kernel)
        scores = numpy.linalg.norm(dst, axis=2)
        score = scores.max()
        target_y,target_x = numpy.unravel_index(scores.argmax(), scores.shape)

        width  = start["w"]
        height = start["h"]

        bounds_first = (int(target_x-width/2), int(target_y-height/2))
        bounds_second = (int(target_x+width/2),int(target_y+height/2))
        cv.rectangle(frame, bounds_first, bounds_second, color, size)
        print(f"\r X: {target_x:7.3f} Y: {target_y:7.3f} W: {width:4} H: {height:4} score: {score:4.2f}", end="")

        cv.imshow("Tracker v0.0.0", frame)
        if cv.waitKey(1) == ord('q'):
            break

    print("")
