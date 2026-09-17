import random
import math
from ackermann_steering import ackermann_wheel_angles

# Use the same constants from your Task 1A file
from ackermann_steering import WHEELBASE, TRACK_WIDTH, WHEEL_OFFSET


def expected_ackermann(delta):

    if abs(delta) < 1e-9:
        return 0.0, 0.0

    half_track = (TRACK_WIDTH / 2.0) + WHEEL_OFFSET

    R = WHEELBASE / math.tan(abs(delta))

    R_inner = R - half_track
    R_outer = R + half_track

    inner_angle = math.atan2(WHEELBASE, R_inner)
    outer_angle = math.atan2(WHEELBASE, R_outer)

    if delta > 0:
        return inner_angle, outer_angle
    else:
        return -outer_angle, -inner_angle


# Test 1000 random angles
total = 1000
passed = 0

for i in range(total):

    delta = random.uniform(-0.522, 0.522)

    actual_left, actual_right = ackermann_wheel_angles(delta)
    expected_left, expected_right = expected_ackermann(delta)

    left_error = abs(float(actual_left) - expected_left)
    right_error = abs(float(actual_right) - expected_right)

    if left_error <= 0.1 and right_error <= 0.1:
        passed += 1
    else:
        print("FAIL")
        print(f"delta = {delta}")
        print(f"Actual   : {actual_left}, {actual_right}")
        print(f"Expected : {expected_left}, {expected_right}")
        print(f"Error    : {left_error}, {right_error}")


print()
print("================================")
print(f"Random tests : {passed}/{total} passed")
print("================================")

if passed == total:
    print("SUCCESS: All random tests passed!")
else:
    print("WARNING: Some random tests failed.")