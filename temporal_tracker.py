#! /usr/bin/env python3
"""
    BSD 3-Clause License

    Copyright (c) 2019, John Drogo

    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice,
          this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.
        * Neither the name of Temporal TemporalTracker nor the names of its contributors
          may be used to endorse or promote products derived from this software
          without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import argparse
import cv2 as cv
import numpy
import time
import signal


class TemporalTracker:
    C_VERSION = "v0.0.3"
    C_KALMAN_STATE_ORDER = 4
    C_MAX_SCORE = 255
    C_MIN_SCORE_THRESHOLD = 10

    # Measured state of the tracked object.
    class State:
        def __init__(self, x, y, w, h, dx, dy):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.dx = dx
            self.dy = dy


    # Initializes the tracker. The start_frame is the first frame for the video. The start_pos is a list of
    # the [x,y] position of the top left point of the target. 
    def __init__(self, start_frame, start_pos):
        # Initialize the tracker to the target's initial position. Assume it is not moving.
        state = TemporalTracker.State(*start_pos+[0,0])

        # Capture the initial kernels.
        self.positive_kernel = self._chip_to_kernel(self._grab_chip(frame, state.x, state.y, state.w, state.h))
        self.negative_kernel = self._grab_neg_kernels(frame, state.x, state.y, state.w, state.h)
        self.prev_state = state

        # Tuning factors for the tracker.

        # Stddev in pixels for the high intensity focus region.
        self.focus_pred_stddev = 16
        # Weight for the focus region compared to regular region.
        self.focus_pred_weight = 0.3
        # Stddev in pixels of regular weights around predicited location.
        self.reg_pred_stddev = 200
        # Learning rate for the kernels.
        self.learning_rate = 0.02
        # Kalman process noise.
        self.process_noise = 5e-3
        # Weights for the negative kernel influcence on correlation score.
        self.negative_kernel_weight = 1

        # Four dimmensional Kalman state yields a 2D model of position and velocity.
        self.kalman = cv.KalmanFilter(TemporalTracker.C_KALMAN_STATE_ORDER, TemporalTracker.C_KALMAN_STATE_ORDER)
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

        # Prime the initial state of the filter.
        self.kalman.processNoiseCov = self.process_noise * numpy.eye(TemporalTracker.C_KALMAN_STATE_ORDER).astype("float32")
        self.kalman.statePre = numpy.array([ [state.x], [state.y], [0], [0] ], numpy.float32)


    @staticmethod
    def _grab_neg_kernels(frame, tl_x, tl_y, w, h):
        # Take advantage of the linearity of convolution to combine the negative kernels.
        neg_kernel = numpy.zeros((h, w, frame.shape[2]))
        num_kernels = 0
        # Scan the region around the target of interest to determine set of negative images.
        # These are averaged together to form a single negative kernel. Leaves on chip size
        # of padding from the target if interest to protect againt imperfect tracks from
        # putting a portion of the target into the negative scores. Currently the algorithm
        # takes four chips, one above, below, left, and right.
        for tl_x_tmp in numpy.array([-2, 2])*w+tl_x:
            for tl_y_tmp in numpy.array([-2, 2])*h+tl_y:
                try:
                    # Grab a negative chip if there is enough data else ignore the chip.
                    neg_kernel += TemporalTracker._chip_to_kernel(TemporalTracker._grab_chip(frame, tl_x_tmp, tl_y_tmp, w, h))
                    num_kernels += 1

                    if numpy.isnan(neg_kernel).any():
                        continue

                except ValueError:
                    continue

        if num_kernels == 0:
            return neg_kernel

        return neg_kernel/num_kernels



    @staticmethod
    # Grabs a chip based on the passed width and height and top left coordinates.
    def _grab_chip(frame, tl_x, tl_y, w, h):
        chip = numpy.zeros((h, w, frame.shape[2]))
        partial_chip  = frame[tl_y:tl_y+h,tl_x:tl_x+w]
        chip[0:partial_chip.shape[0],0:partial_chip.shape[1]] = partial_chip
        return  chip


    @staticmethod
    # Normalizes a passed image chip into a tranformation kernel.
    def _chip_to_kernel(chip):
        chip_max = chip.max()
        return chip.astype(float)/(chip.shape[0] * chip.shape[1] * (chip_max if chip_max != 0 else 1))


    @staticmethod
    # Converts a state object into to the top left and bottom right coordinates of
    # the boundary box.
    def _state_to_boundary(state):
        bounds_first = (int(state.x-state.w/2), int(state.y-state.h/2))
        bounds_second = (int(state.x+state.w/2), int(state.y+state.h/2))
        return bounds_first, bounds_second


    @staticmethod
    # Creates the Gaussian weights as a two dimmensional matrix at the
    # specified location in the video frame.
    def _gen_gaussian_2d(x, y, frame_w, frame_h, stddev):
        x_bound = int(max(min(x, frame_w), 0))
        y_bound = int(max(min(y, frame_h), 0))
        # Size the kernel to include 99% of the distribution.
        blur_radius = 3*stddev
        # Get one dimmensional slice of the Gaussian distribution.
        gaussian_1d = cv.getGaussianKernel(2*blur_radius, stddev)
        # Use vector multiplication of the Gaussian weights to generate 2D matrix.
        gaussian_2d = cv.normalize(gaussian_1d@numpy.transpose(gaussian_1d), None, 0, 1, cv.NORM_MINMAX)
        # Padd the Gaussian to the size of the image based on the desired position.
        padded = numpy.pad(gaussian_2d, ((y_bound, frame_h-y_bound), (x_bound, frame_w-x_bound)))
        return padded[blur_radius:-blur_radius, blur_radius:-blur_radius]


    # Process the frame through the video tracker and return boundary box, scores, and predicted state.
    def process_frame(self, frame):
        state = TemporalTracker.State(0, 0, self.prev_state.w, self.prev_state.h, 0, 0)

        # Kernel should be normalized from 0 to 1. Combine the positive and negative
        # kernels to generate the combined match kernel.
        match_kernel = cv.normalize(self.positive_kernel-self.negative_kernel*self.negative_kernel_weight,
                                    None,
                                    0,
                                    frame.shape[2]/self.positive_kernel.size, cv.NORM_MINMAX)

        # Take the norm of the correlation scores across color channels to
        # generate the final score per pixel.
        scores_raw = numpy.linalg.norm(cv.filter2D(frame, -1, match_kernel), axis=2)

        # Update the Kalman based on the previous measurement.
        self.kalman.correct(numpy.array([[self.prev_state.x], [self.prev_state.y],
                                        [self.prev_state.dx], [self.prev_state.dy] ],
                                        numpy.float32))

        # Predict forward to this measurement.
        predicted_state = self.kalman.predict()

        # Generate Gaussian weights to apply to the search window.
        search_window = max(int((TemporalTracker.C_MAX_SCORE-scores_raw.max())*self.reg_pred_stddev/TemporalTracker.C_MAX_SCORE), 1)
        weights = numpy.ones(scores_raw.shape)
        weights = self._gen_gaussian_2d(predicted_state[0], predicted_state[1],
                                        frame.shape[1], frame.shape[0], search_window)

        # Generate new weights by combining the fine focus and regular distributions.
        weights = (1-self.focus_pred_weight)*weights + self.focus_pred_weight*self._gen_gaussian_2d(predicted_state[0],
                                                                                                    predicted_state[1],
                                                                                                    frame.shape[1],
                                                                                                    frame.shape[0],
                                                                                                    self.focus_pred_stddev)
        scores = scores_raw * weights
        score = scores.max()

        # If there isn't a high score ignore the weights and search the whole image for a match.
        if score < self.C_MIN_SCORE_THRESHOLD:
            scores = scores_raw
            score = scores.max()

        # Scores go from 0 to 255, if after lifting the weights there still isn't a good match coast the track.
        if score < self.C_MIN_SCORE_THRESHOLD:
            state.y = max(min(float(predicted_state[1]), frame.shape[0]), 0)
            state.x = max(min(float(predicted_state[0]), frame.shape[1]), 1)
            state.dx = state.x - self.prev_state.x
            state.dy = state.y - self.prev_state.y
            boundary = self._state_to_boundary(state)

        # Otherwise update the Kalman filter and return the boundary box for the target.
        else:
            state.y,state.x = numpy.unravel_index(scores.argmax(), scores.shape)
            state.dx = state.x - self.prev_state.x
            state.dy = state.y - self.prev_state.y

            boundary = self._state_to_boundary(state)

            # Update the positive and negative kernels based on the learning rate. Can be thought of as a
            # generating combination of weights for correlation with the old kernels verses the new kernels.
            new_chip = self._grab_chip(frame, boundary[0][0], boundary[0][1], self.prev_state.w, self.prev_state.h)
            self.positive_kernel = (1-self.learning_rate)*self.positive_kernel + self.learning_rate*self._chip_to_kernel(new_chip)
            self.negative_kernel = ((1-self.learning_rate)*self.negative_kernel
                                    + self.learning_rate*self._grab_neg_kernels(frame,boundary[0][0],
                                                                                boundary[0][1],
                                                                                self.prev_state.w,
                                                                                self.prev_state.h))
        self.prev_state = state
        return boundary + (score,scores,predicted_state[0], predicted_state[1])
        


# Read frames from the video. OpenCV's data format is BGR or channel zero is blue,
# one is green, and two is red.
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
    C_WINDOW_SPACING = 100
    C_TRACKER_NAME = f"TemporalTracker {TemporalTracker.C_VERSION}"
    C_TRACKER_DETAIL_SIZE = 2

    C_TEST_CASES = {
        "football": {
            "path": "media/Football/football.mp4",
            "start_pos": [310,102,39,50]
        },
        # Interesting case, while easiest to detect the infinite acceleration when
        # the ball hits the edges throws the Kalman off. To get better results the
        # search window can be increased, but this affects performance of real
        # life video where infinite acceleration is less likely.
        "ball": {
            "path": "media/Ball/ball.mp4",
            "start_pos": [0, 0, 40, 40]
        },
        "redteam": {
            "path": "media/RedTeam/redteam.mp4",
            "start_pos": [197, 95, 38, 18]
        },
        # I was curious whether the algorithm was simply locking onto the color red
        # or if was using actual video features. To test this I desaturated the video
        # and saw similar performance to the colored version.
        "redteam_bw": {
            "path": "media/RedTeam/redteam_bw.mp4",
            "start_pos": [197, 95, 38, 18]
        },
        "car": {
            "path": "media/CarScale/car.mp4",
            "start_pos": [29, 170, 22, 16]
        }
    }

    # Parse input arguments for the test setup.
    parser = argparse.ArgumentParser(description="Run temporal tracker on a test video.")
    parser.add_argument("--test_case",
                        choices=list(C_TEST_CASES.keys()),
                        help="Video test case to run through the tracker.",
                        required=True)
    args = parser.parse_args()

    # The initial boundary box for the target of interest. Position is [x,y,w,h]
    # where x and y are the coordinate for top left of the box and w and h
    # are the width and height of the box.
    start_pos = C_TEST_CASES[args.test_case]["start_pos"]
    filename = C_TEST_CASES[args.test_case]["path"]

    print(C_TRACKER_NAME)
    print("")
    print("Press 'q' with one of the windows in focus or Ctrl+C in the terminal to exit")
    print("")
    print("Key:")
    print("    Green box - Tracker bounding box")
    print("    Blue dot - Kalman estimated position")
    print("")

    # Setup script for graceful exiting.
    try:
        should_exit = False
        while not should_exit:
            reader = VideoReader(filename)
            first_frame = True
            for frame in reader.getframes():
                if first_frame:
                    tracker = TemporalTracker(frame, start_pos)
                    cv.waitKey(1000)

                    # Generate initial window layout.
                    norm_to_pixel = C_MAX_PIXEL_VALUE*tracker.positive_kernel.size/tracker.positive_kernel.shape[2]
                    cv.imshow(C_TRACKER_NAME, frame)
                    cv.moveWindow(C_TRACKER_NAME, 0, 0)

                    cv.imshow("Scores", numpy.zeros(frame.shape))
                    cv.moveWindow("Scores", 0, frame.shape[0]+C_WINDOW_SPACING)

                    cv.imshow("Positive Kernel", (cv.normalize(tracker.positive_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
                    cv.moveWindow("Positive Kernel", frame.shape[1]+C_WINDOW_SPACING, 0)

                    cv.imshow("Negative Kernel", (cv.normalize(tracker.negative_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
                    cv.moveWindow("Negative Kernel", frame.shape[1]+C_WINDOW_SPACING, tracker.positive_kernel.shape[0]+C_WINDOW_SPACING)

                    first_frame = False
                    continue

                # Process a frame through the tracker, draw the boundary box and predicted state.
                bounds_first,bounds_second,score,scores,pred_x,pred_y = tracker.process_frame(frame)
                cv.rectangle(frame, bounds_first, bounds_second, C_TRACKER_COLOR, C_TRACKER_DETAIL_SIZE)
                cv.circle(frame, (pred_x, pred_y), C_TRACKER_DETAIL_SIZE, C_PREDICT_COLOR, C_TRACKER_DETAIL_SIZE)

                # Print tracker statistics.
                print(f"\r X: {tracker.prev_state.x:7} Y: {tracker.prev_state.y:7} "+
                        f"W: {tracker.prev_state.w:4} H: {tracker.prev_state.h:4} score: {score:4.2f}", end="")

                # Normalize the scores to pixel values to show with the imshow command.
                normalized_scores = scores.astype("uint8")

                # Display tracker diagnostic information.
                cv.imshow(C_TRACKER_NAME, frame)
                cv.imshow("Scores", normalized_scores)
                cv.imshow("Positive Kernel", (cv.normalize(tracker.positive_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
                cv.imshow("Negative Kernel", (cv.normalize(tracker.negative_kernel, None, C_MAX_PIXEL_VALUE, 0, cv.NORM_MINMAX).astype("uint8")))
                if cv.waitKey(1) == ord('q'):
                    should_exit = True
                    break

    except KeyboardInterrupt:
        pass

    print("")
    print("Exiting...")
