'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         lane_detection.py
# Functions:        detect_lane
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_lane(frame):
    '''
    Purpose:
    ---
    Detect the lane in a single frame and report where the centre of the lane
    is, and which of the two lanes the vehicle is currently in.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        A single BGR frame read from the video, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "center_x" : int,   x-pixel of the lane centre in this frame,
                                or -1 if the lane could not be found
            "lane"     : str,   "left", "right" or "unknown"
        }

    Example call:
    ---
    result = detect_lane(frame)

    COORDINATE SYSTEM:
    ---
    `center_x` is an absolute pixel column in the frame AS RECEIVED - the
    dataset's own resolution, 640x480. It is compared against a ground truth
    measured in those pixels, so it only means anything in them.

    You may resize, crop or warp all you like inside this function, but scale
    the answer back before returning it. A centre found in a 320x240 copy is
    half the value it should be, and a centre read off a bird's-eye view is in
    warped coordinates, not frame ones - map the point back through the inverse
    of your transform. Do not re-encode or resize the clip files themselves.

    NOTE:
    ---
    This function must ONLY compute and return the result.
    Do not call cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from
    inside it. All visualisation and debugging output belongs outside this
    function - see draw_overlay() and process_video() below.
    '''

    center_x = -1
    lane = LANE_UNKNOWN

    #################### ADD YOUR CODE HERE ####################
    # 1. Isolate the lane markings in `frame`
    # 2. Work out which two markings bracket the vehicle
    # 3. Compute the x-pixel of the lane centre   ->  center_x
    # 4. Decide which lane the vehicle is in      ->  lane

    height, width = frame.shape[:2]

    # ---------------------------------------------------------
    # Convert BGR image to HSV
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---------------------------------------------------------
    # Detect YELLOW road markings
    # ---------------------------------------------------------
    yellow_lower = np.array([15, 80, 80], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    # ---------------------------------------------------------
    # Detect WHITE road markings
    # ---------------------------------------------------------
    white_lower = np.array([0, 0, 160], dtype=np.uint8)
    white_upper = np.array([180, 100, 255], dtype=np.uint8)

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )

    # ---------------------------------------------------------
    # Region of Interest
    #
    # Ignore the upper part of the image because the road
    # markings are mainly useful in the lower road region.
    # ---------------------------------------------------------
    roi_top = int(height * 0.40)

    yellow_mask[:roi_top, :] = 0
    white_mask[:roi_top, :] = 0

    # ---------------------------------------------------------
    # Remove small noise
    # ---------------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # ---------------------------------------------------------
    # Helper function
    #
    # Estimates the x-coordinate of a lane marking by looking
    # at several horizontal bands instead of relying on one
    # particular row.
    # ---------------------------------------------------------
    # NEW: per-row (y -> x) points kept for the paired-row refinement below.
    _rows = {}

    def estimate_marking_x(mask, x_start, x_end, _key=None):

        y_start = int(height * 0.55)
        y_end = int(height * 0.96)

        points = []
        _ys = []

        for y in range(y_start, y_end, 6):

            ya = max(0, y - 3)
            yb = min(height, y + 4)

            band = mask[ya:yb, x_start:x_end]

            xs = np.where(
                np.any(band > 0, axis=0)
            )[0]

            if len(xs) == 0:
                continue

            xs = xs + x_start

            # Median is robust against isolated pixels.
            points.append(_sorted_median(xs))
            _ys.append(y)

        if _key is not None:
            _rows[_key] = dict(zip(_ys, points))

        if len(points) < 4:
            return None

        points = np.asarray(points, dtype=np.float32)

        # Remove outliers.
        median = float(np.median(points))
        deviation = np.abs(points - median)

        mad = float(np.median(deviation))

        if mad > 1.0:
            threshold = max(15.0, 3.0 * mad)
            points = points[deviation <= threshold]

        if len(points) < 4:
            return None

        return float(np.median(points))

    # ---------------------------------------------------------
    # Vehicle reference position
    #
    # The camera is approximately centred in the 640 px frame.
    # ---------------------------------------------------------
    vehicle_x = width / 2.0

    # ---------------------------------------------------------
    # Detect yellow boundaries
    # ---------------------------------------------------------

    yellow_left_x = estimate_marking_x(
        yellow_mask,
        0,
        int(vehicle_x),
        "yl"
    )

    yellow_right_x = estimate_marking_x(
        yellow_mask,
        int(vehicle_x),
        width,
        "yr"
    )

    # ---------------------------------------------------------
    # Detect white centre divider
    #
    # Search separately on the two sides of the vehicle.
    # ---------------------------------------------------------

    white_left_x = estimate_marking_x(
        white_mask,
        0,
        int(vehicle_x),
        "wl"
    )

    white_right_x = estimate_marking_x(
        white_mask,
        int(vehicle_x),
        width,
        "wr"
    )

    # ---------------------------------------------------------
    # Determine LEFT lane
    #
    # If the dashed white divider is to the right of the
    # vehicle, the vehicle is in the LEFT lane.
    #
    # Lane boundaries:
    #
    # yellow left edge ---- lane ---- white divider
    # ---------------------------------------------------------

    if white_right_x is not None:

        if white_right_x > vehicle_x + 5:

            if yellow_left_x is not None:

                if yellow_left_x < white_right_x:

                    center_x = int(
                        round(
                            (yellow_left_x + white_right_x) / 2.0
                        )
                    )

                    lane = LANE_LEFT

    # ---------------------------------------------------------
    # Determine RIGHT lane
    #
    # If the dashed white divider is to the left of the
    # vehicle, the vehicle is in the RIGHT lane.
    #
    # Lane boundaries:
    #
    # white divider ---- lane ---- yellow right edge
    # ---------------------------------------------------------

    if center_x == -1:

        if white_left_x is not None:

            if white_left_x < vehicle_x - 5:

                if yellow_right_x is not None:

                    if white_left_x < yellow_right_x:

                        center_x = int(
                            round(
                                (white_left_x + yellow_right_x) / 2.0
                            )
                        )

                        lane = LANE_RIGHT

    # ---------------------------------------------------------
    # CLAUDE PATCH
    #
    # Look for a coherent far-right white line beyond the
    # yellow boundary. This is only allowed to override the
    # baseline when the geometry passes several strict tests.
    # ---------------------------------------------------------

    _from_third = False

    if lane == LANE_RIGHT and yellow_left_x is None:

        _c = _third_line_left_center(
            yellow_mask,
            white_mask,
            height,
            width,
            vehicle_x
        )

        if _c is not None:

            center_x = _c
            lane = LANE_LEFT
            _from_third = True

    # ---------------------------------------------------------
    # HYBRID CENTER REFINEMENT
    #
    # Keep the lane classification above completely unchanged.
    # When the two boundaries for the already-selected lane can
    # be fitted reliably, estimate their midpoint at a fixed,
    # empirically stable road row. If the geometry is unavailable,
    # retain the existing center_x exactly.
    # ---------------------------------------------------------

    if lane == LANE_LEFT:
        _hybrid_center = _fixed_row_center(
            yellow_mask,
            white_mask,
            height,
            width,
            vehicle_x,
            290.0,
            0,
            int(vehicle_x),
            int(vehicle_x),
            width
        )

        if _hybrid_center is not None:
            center_x = _hybrid_center
        else:
            _fallback_center = _hood_fallback_center(
                yellow_mask,
                white_mask,
                height,
                width,
                vehicle_x,
                290.0,
                False,
                center_x
            )
            if _fallback_center is not None:
                center_x = _fallback_center
            elif not _from_third:
                _paired = _paired_from_rows(_rows.get("yl"), _rows.get("wr"), width)
                if _paired is not None:
                    center_x = _paired

    elif lane == LANE_RIGHT:
        _hybrid_center = _fixed_row_center(
            white_mask,
            yellow_mask,
            height,
            width,
            vehicle_x,
            275.0,
            0,
            int(vehicle_x),
            int(vehicle_x),
            width
        )

        if _hybrid_center is not None:
            center_x = _hybrid_center
        else:
            _fallback_center = _hood_fallback_center(
                white_mask,
                yellow_mask,
                height,
                width,
                vehicle_x,
                275.0,
                True,
                center_x
            )
            if _fallback_center is not None:
                center_x = _fallback_center
            elif not _from_third:
                _paired = _paired_from_rows(_rows.get("wl"), _rows.get("yr"), width)
                if _paired is not None:
                    center_x = _paired

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if center_x < 0 or center_x >= width:

        center_x = -1
        lane = LANE_UNKNOWN

    ############################################################


    return {"center_x": center_x, "lane": lane}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_lane() - the evaluation script only
# ever calls that one function. List them in the file header too.
# ------------------------------------------------------------------


def _sorted_median(xs):
    """Median of an ascending 1-D integer array (np.where output).
    Exactly equal to float(np.median(xs)) but without numpy's per-call overhead."""
    n = len(xs)
    m = n // 2
    if n % 2:
        return float(xs[m])
    return (float(xs[m - 1]) + float(xs[m])) / 2.0


def _band_points(mask, x_lo, x_hi, height):
    pts = []

    for y in range(
        int(height * 0.55),
        int(height * 0.96),
        6
    ):

        lo = max(
            0,
            int(x_lo(y))
        )

        hi = int(x_hi)

        if hi - lo < 3:
            continue

        band = mask[
            max(0, y - 3):min(height, y + 4),
            lo:hi
        ]

        xs = np.where(
            np.any(band > 0, axis=0)
        )[0]

        if len(xs):

            pts.append(
                (
                    float(y),
                    _sorted_median(xs) + lo
                )
            )

    return pts


def _fit_x_of_y(pts, tol, min_pts):

    if len(pts) < min_pts:
        return None

    arr = np.asarray(
        pts,
        dtype=np.float64
    )

    ys = arr[:, 0]
    xs = arr[:, 1]

    while len(ys) >= min_pts:

        b, a = np.polyfit(
            ys,
            xs,
            1
        )

        res = np.abs(
            xs - (a + b * ys)
        )

        worst = int(
            np.argmax(res)
        )

        if res[worst] <= tol:
            return a, b

        ys = np.delete(
            ys,
            worst
        )

        xs = np.delete(
            xs,
            worst
        )

    return None


def _fixed_row_center(
    mask_a,
    mask_b,
    height,
    width,
    vehicle_x,
    y_ref,
    a_lo,
    a_hi,
    b_lo,
    b_hi
):
    """
    Fit the two visible lane-boundary markings and evaluate both
    at a fixed image row. This is used only to improve center_x;
    lane classification is never changed by this helper.
    """
    fit_a = _fit_x_of_y(
        _band_points(
            mask_a,
            lambda y: a_lo,
            a_hi,
            height
        ),
        4.0,
        6
    )

    fit_b = _fit_x_of_y(
        _band_points(
            mask_b,
            lambda y: b_lo,
            b_hi,
            height
        ),
        4.0,
        6
    )

    if fit_a is None or fit_b is None:
        return None

    aa, ba = fit_a
    ab, bb = fit_b

    xa = aa + ba * y_ref
    xb = ab + bb * y_ref

    c = int(round((xa + xb) / 2.0))

    if 0 <= c < width:
        return c

    return None


def _fit_x_of_y_rows(pts, tol, min_pts):
    """Same greedy line fit as _fit_x_of_y, but also returns surviving rows."""
    if len(pts) < min_pts:
        return None

    arr = np.asarray(pts, dtype=np.float64)
    ys = arr[:, 0]
    xs = arr[:, 1]

    while len(ys) >= min_pts:
        b, a = np.polyfit(ys, xs, 1)
        res = np.abs(xs - (a + b * ys))
        worst = int(np.argmax(res))

        if res[worst] <= tol:
            return a, b, ys.copy()

        ys = np.delete(ys, worst)
        xs = np.delete(xs, worst)

    return None


def _hood_fallback_center(
    mask_a,
    mask_b,
    height,
    width,
    vehicle_x,
    y_ref,
    white_is_a,
    current_center
):
    """
    Recover center_x when the live fixed-row fit fails because the white
    marking contains hood/foreground contamination. The geometry is the
    same candidate-A geometry used in the public diagnostic: white rows
    at y >= 420 are removed, then strict support, vanishing-row, lane-width,
    and shift gates are applied. This helper never changes the lane label.
    """
    if not (0 <= current_center < width):
        return None

    vx = int(vehicle_x)

    # mask_a is always the left-side boundary and mask_b the right-side
    # boundary, matching the arguments used by _fixed_row_center().
    pts_a = _band_points(
        mask_a,
        lambda y: 0,
        vx,
        height
    )
    pts_b = _band_points(
        mask_b,
        lambda y: vx,
        width,
        height
    )

    if white_is_a:
        white_pts = [p for p in pts_a if p[0] < 420.0]
        yellow_pts = pts_b
    else:
        yellow_pts = pts_a
        white_pts = [p for p in pts_b if p[0] < 420.0]

    white_fit = _fit_x_of_y_rows(white_pts, 4.0, 6)
    yellow_fit = _fit_x_of_y_rows(yellow_pts, 4.0, 6)

    if white_fit is None or yellow_fit is None:
        return None

    aw, bw, white_ys = white_fit
    ay, by, yellow_ys = yellow_fit

    xw = aw + bw * y_ref
    xy = ay + by * y_ref

    candidate = int(round((xw + xy) / 2.0))

    if not (0 <= candidate < width):
        return None

    # Support: neither fitted line may be extrapolated more than 24 px
    # beyond the surviving observations at the reference row.
    white_extrap = max(
        abs(float(y_ref) - float(np.min(white_ys))),
        abs(float(y_ref) - float(np.max(white_ys)))
    )
    yellow_extrap = max(
        abs(float(y_ref) - float(np.min(yellow_ys))),
        abs(float(y_ref) - float(np.max(yellow_ys)))
    )

    if max(white_extrap, yellow_extrap) > 24.0:
        return None

    # The two fitted lines should meet in the upper image at a plausible
    # vanishing row.
    denom = by - bw
    if abs(denom) < 1e-9:
        return None

    vp = (aw - ay) / denom

    if not (30.0 <= vp <= 100.0):
        return None

    # Public-set reference width bands measured from accurate live-hybrid
    # frames. They are intentionally lane-specific.
    lane_width = abs(xw - xy)

    if white_is_a:
        # RIGHT lane: white is left, yellow is right.
        if not (292.0 <= lane_width <= 323.0):
            return None
    else:
        # LEFT lane: yellow is left, white is right.
        if not (306.0 <= lane_width <= 370.0):
            return None

    # Do not move the center by more than the diagnostic shift gate.
    if abs(float(candidate) - float(current_center)) > 50.0:
        return None

    return candidate


def _third_line_left_center(
    yellow_mask,
    white_mask,
    height,
    width,
    vehicle_x
):

    vx = int(vehicle_x)

    # ---------------------------------------------------------
    # Fit the yellow marking on the right
    # ---------------------------------------------------------

    yfit = _fit_x_of_y(
        _band_points(
            yellow_mask,
            lambda y: vx,
            width,
            height
        ),
        4.0,
        8
    )

    if yfit is None:
        return None

    ay, by = yfit

    if not (0.6 <= by <= 1.5):
        return None

    # ---------------------------------------------------------
    # Fit the left white marking
    # ---------------------------------------------------------

    lfit = _fit_x_of_y(
        _band_points(
            white_mask,
            lambda y: 0,
            vx - 15,
            height
        ),
        4.0,
        6
    )

    if lfit is None:
        return None

    al, bl = lfit

    if bl > -0.4:
        return None

    # ---------------------------------------------------------
    # Search for another white line at least 40 px
    # beyond the fitted yellow line.
    # ---------------------------------------------------------

    rpts = _band_points(
        white_mask,
        lambda y: ay + by * y + 40,
        width,
        height
    )

    if len(rpts) < 2:
        return None

    y0, x0 = rpts[0]
    y1, x1 = rpts[-1]

    if y1 - y0 < 30:
        return None

    slope = (
        (x1 - x0) /
        (y1 - y0)
    )

    if not (
        by + 0.15 < slope <= 3.0
    ):
        return None

    # ---------------------------------------------------------
    # Check that the far-right white points are approximately
    # collinear.
    # ---------------------------------------------------------

    for y, x in rpts:

        expected_x = (
            x0 +
            slope * (y - y0)
        )

        if abs(x - expected_x) > 12:
            return None

    # ---------------------------------------------------------
    # Evaluate both white lines at the same y reference.
    # ---------------------------------------------------------

    y_ref = float(
        np.mean(
            [p[0] for p in rpts]
        )
    )

    xr = float(
        np.mean(
            [p[1] for p in rpts]
        )
    )

    xl = (
        al +
        bl * y_ref
    )

    # Left white line must be clearly left
    # of the vehicle.
    if xl > vx - 40:
        return None

    # ---------------------------------------------------------
    # New lane centre
    # ---------------------------------------------------------

    c = int(
        round(
            (xl + xr) / 2.0
        )
    )

    if 0 <= c < width:
        return c

    return None


def _paired_from_rows(rows_left, rows_right, width):
    """
    Row-consistent lane centre from per-row boundary x values that
    estimate_marking_x() has already measured (no extra image work).

    Both boundaries are paired on the SAME image rows, so the dash-phase
    row mismatch of the plain median baseline cancels. The result is only
    trusted when those rows lie close to the (empirical) reference rows,
    i.e. mean paired row <= 330, so the midpoint is not carried far along
    the lane centre-line from where it was measured.
    """
    if not rows_left or not rows_right:
        return None

    ys = sorted(y for y in rows_left if y in rows_right)
    if len(ys) < 4:
        return None

    ys = np.asarray(ys, dtype=np.float64)
    mids = np.asarray(
        [0.5 * (rows_left[int(y)] + rows_right[int(y)]) for y in ys],
        dtype=np.float64
    )

    med = float(np.median(mids))
    dev = np.abs(mids - med)
    mad = float(np.median(dev))
    if mad > 1.0:
        keep = dev <= max(15.0, 3.0 * mad)
        ys = ys[keep]
        mids = mids[keep]

    if len(mids) < 4:
        return None

    if round(float(ys.mean()), 1) > 330.0:
        return None

    c = int(round(float(np.median(mids))))
    if 0 <= c < width:
        return c
    return None


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]      whatever detect_lane() returned
    `frame_index` :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :  [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video file
    `show` :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video that was processed
    `results` :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :   [ list ]    the raw command-line arguments

    Returns:
    ---
    `videos` :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()
