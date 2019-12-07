import cv2 as cv
import numpy
import time


class Tracker:
    class State:
        def __init__(self, x, y, w, h):
            self.x = x
            self.y = y
            self.w = w
            self.h = h

    def __init__(self, start_frame, start_pos):
        state = Tracker.State(*start_pos)
        chip = frame[state.y:state.y+state.h, state.x:state.x+state.w]
        kernel = chip.astype(float)/(chip.shape[0] * chip.shape[1] * chip.max())
        self.positive_kernel = kernel
        self.prev_state = state

    @staticmethod
    def _state_to_boundary(state):
        bounds_first = (int(state.x-state.w/2), int(state.y-state.h/2))
        bounds_second = (int(state.x+state.w/2), int(state.y+state.h/2))
        return bounds_first, bounds_second

    def process_frame(self, frame):
        state = Tracker.State(0, 0, self.prev_state.w, self.prev_state.h)

        dst = cv.filter2D(frame, ddepth, self.positive_kernel)
        scores = numpy.linalg.norm(dst, axis=2)
        score = scores.max()
        state.y,state.x = numpy.unravel_index(scores.argmax(), scores.shape)

        self.prev_state = state
        return self._state_to_boundary(state) + (score,scores,)
        


class VideoReader:
    def __init__(self, filepath):
        self.capture = cv.VideoCapture(cv.samples.findFileOrKeep(filepath))
        if not self.capture.isOpened:
            raise ValueError("Open failure.")


    def getframes(self):
        while True:
            ret,frame = self.capture.read()
            if ret is False:
                break

            yield frame



if __name__ == "__main__":
    C_MAX_PIXEL_VALUE = 255
    color = (0, C_MAX_PIXEL_VALUE, 0)
    size = 2
    ddepth = -1
    start_pos = [310,102,39,50]

    reader = VideoReader("media/Football/football.mp4")
    first_frame = True
    for frame in reader.getframes():
        if first_frame:
            tracker = Tracker(frame, start_pos)
            cv.waitKey(1000)
            norm_to_pixel = C_MAX_PIXEL_VALUE*tracker.positive_kernel.size/tracker.positive_kernel.shape[2]
            cv.imshow("Initial frame", (cv.normalize(tracker.positive_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
            first_frame = False
            continue

        bounds_first,bounds_second,score,scores = tracker.process_frame(frame)
        cv.rectangle(frame, bounds_first, bounds_second, color, size)
        print(f"\r X: {tracker.prev_state.x:7.3f} Y: {tracker.prev_state.y:7.3f} W: {tracker.prev_state.w:4} H: {tracker.prev_state.h:4} score: {score:4.2f}", end="")

        cv.imshow("Tracker v0.0.1", frame)
        cv.imshow("Matches", cv.normalize(scores, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8"))
        if cv.waitKey(1) == ord('q'):
            break

    print("\nVideo ended")
    print("")

    #test = cv.getGaussianKernel(16, 7); cv.imshow("Gaussian", (test@numpy.transpose(test)/(test.max()**2)*255).astype("uint8")); cv.waitKey(1)
