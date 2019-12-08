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
        self.positive_kernel = self._chip_to_kernel(self._grab_chip(frame, state.x, state.y, state.w, state.h))

        self.negative_kernel = self._grab_neg_kernels(frame, state.x, state.y, state.w, state.h)
        self.prev_state = state

        self.focus_pred_stddev = 16
        self.focus_pred_weight = 0.3
        self.reg_pred_stddev = 200
        self.learning_rate = 0.02
        self.process_noise = 5e-3
        self.negative_kernel_weight = 1

        self.kalman = cv.KalmanFilter(4, 4)
        self.kalman.transitionMatrix = numpy.array([
            # State is [ x y dx dy ]' as a column vector.
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], numpy.float32)
        self.kalman.measurementMatrix = numpy.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], numpy.float32)
        self.kalman.processNoiseCov = self.process_noise * numpy.eye(4).astype("float32")
        self.kalman.statePre = numpy.array([ [state.x], [state.y], [0], [0] ], numpy.float32)


    @staticmethod
    def _grab_neg_kernels(frame, tl_x, tl_y, w, h):
        # Take advantage of the linearity of convolution to combine the kernels.
        neg_kernel = numpy.zeros((h, w, 3))
        num_kernels = 0
        for tl_x_tmp in numpy.array([-2, 2])*w+tl_x:
            for tl_y_tmp in numpy.array([-2, 2])*h+tl_y:
                try:
                    neg_kernel += Tracker._chip_to_kernel(Tracker._grab_chip(frame, tl_x_tmp, tl_y_tmp, w, h))
                    num_kernels += 1

                    if numpy.isnan(neg_kernel).any():
                        continue

                except ValueError:
                    continue

        if num_kernels == 0:
            return neg_kernel

        return neg_kernel/num_kernels



    @staticmethod
    def _grab_chip(frame, tl_x, tl_y, w, h):
        chip = numpy.zeros((h, w, 3))
        partial_chip  = frame[tl_y:tl_y+h,tl_x:tl_x+w]
        chip[0:partial_chip.shape[0],0:partial_chip.shape[1]] = partial_chip
        return  chip


    @staticmethod
    def _chip_to_kernel(chip):
        chip_max = chip.max()
        return chip.astype(float)/(chip.shape[0] * chip.shape[1] * (chip_max if chip_max != 0 else 1))


    @staticmethod
    def _state_to_boundary(state):
        bounds_first = (int(state.x-state.w/2), int(state.y-state.h/2))
        bounds_second = (int(state.x+state.w/2), int(state.y+state.h/2))
        return bounds_first, bounds_second


    @staticmethod
    def _gen_gaussian_2d(x, y, frame_w, frame_h, stddev):
        x_bound = int(max(min(x, frame_w), 0))
        y_bound = int(max(min(y, frame_h), 0))
        # Size the kernel to include 99% of the distribution.
        blur_radius = 3*stddev
        gaussian_1d = cv.getGaussianKernel(2*blur_radius, stddev)
        gaussian_2d = cv.normalize(gaussian_1d@numpy.transpose(gaussian_1d), None, 0, 1, cv.NORM_MINMAX)
        padded = numpy.pad(gaussian_2d, ((y_bound, frame_h-y_bound), (x_bound, frame_w-x_bound)))
        return padded[blur_radius:-blur_radius, blur_radius:-blur_radius]


    def process_frame(self, frame):
        state = Tracker.State(0, 0, self.prev_state.w, self.prev_state.h, 0, 0)

        # Kernel should be normalized from 0 to 1.
        match_kernel = cv.normalize(self.positive_kernel-self.negative_kernel*self.negative_kernel_weight, None, 0, 3/self.positive_kernel.size, cv.NORM_MINMAX)#self.positive_kernel#((self.positive_kernel - self.negative_kernel)/2) + 0.5/self.positive_kernel.size
        score_tmp = cv.filter2D(frame, ddepth, match_kernel)
        scores = numpy.linalg.norm(score_tmp, axis=2)

        # Update the Kalman based on the previous measurement.
        self.kalman.correct(numpy.array([[self.prev_state.x], [self.prev_state.y], [self.prev_state.dx], [self.prev_state.dy] ], numpy.float32))
        # Predict forward to this measurement.
        predicted_state = self.kalman.predict()

        search_window = max(int((255-scores.max())*self.reg_pred_stddev/255), 1)
        weights = numpy.ones(scores.shape)
        weights = self._gen_gaussian_2d(predicted_state[0], predicted_state[1], frame.shape[1], frame.shape[0], search_window)
        weights = (1-self.focus_pred_weight)*weights + self.focus_pred_weight*self._gen_gaussian_2d(predicted_state[0], predicted_state[1], frame.shape[1], frame.shape[0], self.focus_pred_stddev)
        scores = scores * weights

        score = scores.max()

        if score < 10:
            scores = numpy.linalg.norm(score_tmp, axis=2)
            score = scores.max()


        # Scores go from 0 to 255
        if score < 10:
            state.y = max(min(float(predicted_state[1]), frame.shape[0]), 0)
            state.x = max(min(float(predicted_state[0]), frame.shape[1]), 1)
            state.dx = state.x - self.prev_state.x
            state.dy = state.y - self.prev_state.y
            boundary = self._state_to_boundary(state)
        else:
            state.y,state.x = numpy.unravel_index(scores.argmax(), scores.shape)
            #state.y = (state.y + max(min(float(predicted_state[1]), frame.shape[0]), 1))/2
            #state.x = (state.x + max(min(float(predicted_state[0]), frame.shape[1]), 1))/2
            state.dx = state.x - self.prev_state.x
            state.dy = state.y - self.prev_state.y

            boundary = self._state_to_boundary(state)
            new_chip = self._grab_chip(frame, boundary[0][0], boundary[0][1], self.prev_state.w, self.prev_state.h)
            self.positive_kernel = (1-self.learning_rate)*self.positive_kernel + self.learning_rate*self._chip_to_kernel(new_chip)
            self.negative_kernel = (1-self.learning_rate)*self.negative_kernel + self.learning_rate*self._grab_neg_kernels(frame, boundary[0][0], boundary[0][1], self.prev_state.w, self.prev_state.h)

        self.prev_state = state

        return boundary + (score,scores,predicted_state[0], predicted_state[1])
        


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
    C_TRACKER_COLOR = (0, C_MAX_PIXEL_VALUE, 0)
    C_PREDICT_COLOR = (C_MAX_PIXEL_VALUE, 0, 0)
    size = 2
    ddepth = -1

    while True:
        #reader = VideoReader("media/ball.mp4")
        #start_pos = [0, 0, 40, 40]
        #reader = VideoReader("media/Football/football.mp4")
        #start_pos = [310,102,39,50]
        #reader = VideoReader("media/RedTeam/redteam.mp4")
        #start_pos = [197, 95, 38, 18]
        #reader = VideoReader("media/CarScale/car.mp4")
        #start_pos = [29, 170, 22, 16]
        first_frame = True
        for frame in reader.getframes():
            if first_frame:
                tracker = Tracker(frame, start_pos)
                padding = 100
                cv.waitKey(1000)
                norm_to_pixel = C_MAX_PIXEL_VALUE*tracker.positive_kernel.size/tracker.positive_kernel.shape[2]
                cv.imshow("Tracker v0.0.2", frame)
                cv.moveWindow("Tracker v0.0.2", 0, 0)

                cv.imshow("Scores", numpy.zeros(frame.shape))
                cv.moveWindow("Scores", 0, frame.shape[0]+padding)

                cv.imshow("Positive Kernel", (cv.normalize(tracker.positive_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
                cv.moveWindow("Positive Kernel", frame.shape[1]+padding, 0)

                cv.imshow("Negative Kernel", (cv.normalize(tracker.negative_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
                cv.moveWindow("Negative Kernel", frame.shape[1]+padding, tracker.positive_kernel.shape[0]+padding)

                first_frame = False
                continue

            bounds_first,bounds_second,score,scores,pred_x,pred_y = tracker.process_frame(frame)
            cv.rectangle(frame, bounds_first, bounds_second, C_TRACKER_COLOR, size)
            cv.circle(frame, (pred_x, pred_y), size, C_PREDICT_COLOR, size)

            normalized_scores = scores.astype("uint8")

            print(f"\r X: {tracker.prev_state.x:7.3f} Y: {tracker.prev_state.y:7.3f} W: {tracker.prev_state.w:4} H: {tracker.prev_state.h:4} score: {score:4.2f}", end="")

            cv.imshow("Tracker v0.0.2", frame)
            cv.imshow("Scores", normalized_scores)
            cv.imshow("Positive Kernel", (cv.normalize(tracker.positive_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
            cv.imshow("Negative Kernel", (cv.normalize(tracker.negative_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
            if cv.waitKey(1) == ord('q'):
                break

        print("\nVideo ended")
        print("")
