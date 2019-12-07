import cv2 as cv
import numpy
import time


class Tracker:
    class State:
        def __init__(self, x, y, w, h, dx, dy):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.dx = dx
            self.dy = dy


    def __init__(self, start_frame, start_pos):
        state = Tracker.State(*start_pos+[0,0])
        chip = frame[state.y:state.y+state.h, state.x:state.x+state.w]
        kernel = chip.astype(float)/(chip.shape[0] * chip.shape[1] * chip.max())
        self.positive_kernel = kernel
        self.prev_state = state

        self.kalman = cv.KalmanFilter(4, 4)
        self.kalman.transitionMatrix = numpy.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], numpy.float32)
        self.kalman.measurementMatrix = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], numpy.float32)
        self.kalman.processNoiseCov = 1e-5 * numpy.eye(4).astype("float32")
        self.kalman.statePre = numpy.array([ [state.x], [state.y], [0], [0] ], numpy.float32)
        #self.kalman.measurementNoiseCov = 0.1 * np.ones((2, 1))
        #self.kalman.errorCovPost = 1. * np.ones((2, 2))


    @staticmethod
    def _state_to_boundary(state):
        bounds_first = (int(state.x-state.w/2), int(state.y-state.h/2))
        bounds_second = (int(state.x+state.w/2), int(state.y+state.h/2))
        return bounds_first, bounds_second


    @staticmethod
    def _gen_gaussian_2d(x, y, frame_w, frame_h, stddev):
        # Size the kernel to include 99% of the distribution.
        blur_radius = 3*stddev
        gaussian_1d = cv.getGaussianKernel(2*blur_radius, stddev)
        gaussian_2d = cv.normalize(gaussian_1d@numpy.transpose(gaussian_1d), None, 0, 1, cv.NORM_MINMAX)
        padded = numpy.pad(gaussian_2d, ((y, frame_h-y), (x, frame_w-x)))
        return padded[blur_radius:-blur_radius, blur_radius:-blur_radius]


    def process_frame(self, frame):
        state = Tracker.State(0, 0, self.prev_state.w, self.prev_state.h, 0, 0)

        dst = cv.filter2D(frame, ddepth, self.positive_kernel)

        # Update the Kalman based on the previous measurement.
        self.kalman.correct(numpy.array([[self.prev_state.x], [self.prev_state.y], [self.prev_state.dx], [self.prev_state.dy] ], numpy.float32))
        # Predict forward to this measurement.
        predicted_state = self.kalman.predict()

        weights = self._gen_gaussian_2d(max(int(predicted_state[0]), 0), max(int(predicted_state[1]), 0), frame.shape[1], frame.shape[0], 32)
        scores = cv.normalize(numpy.linalg.norm(dst, axis=2)*weights, None, 0, 1, cv.NORM_MINMAX)
        print(predicted_state, self.kalman.errorCovPost)

        score = scores.max()
        state.y,state.x = numpy.unravel_index(scores.argmax(), scores.shape)
        state.dx = state.x - self.prev_state.x
        state.dy = state.y - self.prev_state.y

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

    while True:
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
            #print(f"\r X: {tracker.prev_state.x:7.3f} Y: {tracker.prev_state.y:7.3f} W: {tracker.prev_state.w:4} H: {tracker.prev_state.h:4} score: {score:4.2f}", end="")

            cv.imshow("Tracker v0.0.1", frame)
            cv.imshow("Matches", cv.normalize(scores, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8"))
            if cv.waitKey(1) == ord('q'):
                break

        print("\nVideo ended")
        print("")
