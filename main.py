import numpy as np
import cv2


def get_curve(img, left_fit_cr, right_fit_cr):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int)/2
    center = (car_pos - lane_center_position)
    return left_curverad, right_curverad, center


def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    canny = cv2.Canny(img, 1000, 10)
    return canny


def curve(img, coords):
    x_mid = img.shape[1]/2

    left = []
    right = []
    for i in coords:
        if i[1] < x_mid:
            left.append(i)
        else:
            right.append(i)

    left_y, left_x = map(list, zip(*left))
    right_y, right_x = map(list, zip(*right))

    left_curve1 = np.polyfit(left_x, left_y, 2)
    left_curve = np.poly1d(left_curve1)
    draw_x = np.linspace(0, img.shape[0] - 1, img.shape[0])
    draw_y = left_curve(draw_x)  # evaluate the polynomial

    right_curve1 = np.polyfit(right_y, right_x, 2)
    right_curve = np.poly1d(right_curve1)
    draw_x2 = np.linspace(0, img.shape[0]-1, img.shape[0])
    draw_y2 = right_curve(draw_x2)  # evaluate the polynomial

    l_rad, r_rad, center = get_curve(img, left_curve1, right_curve1)
    print("CENTER :-", center)
    if (abs(l_rad)-abs(r_rad)) > 0:
        print("LEFT :-", abs(l_rad)-abs(r_rad))
    else:
        print("RIGHT :-", abs(l_rad)-abs(r_rad))
    print("*************")
    draw_points = np.asarray([draw_x, draw_y]).T.astype(np.int32)  # needs to be int32 and transposed
    cv2.polylines(img, [draw_points], False, (0, 0, 255), 100)  # args: image, points, closed, color

    draw_points2 = np.asarray([draw_y2, draw_x2]).T.astype(np.int32)  # needs to be int32 and transposed
    cv2.polylines(img, [draw_points2], False, (0, 0, 255),  100)  # args: image, points, closed, color

    return img

def draw_lines(img, lines):
    # line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                print(int(x1), int(y1), int(x2), int(y2))
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 10)
    return img

def front_top(img, M):
    size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

def top_front(img, M_inv):
    size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M_inv, size, flags=cv2.INTER_LINEAR)

# (hMin = 1 , sMin = 0, vMin = 160), (hMax = 100 , sMax = 255, vMax = 255) ===== Yellow
# (hMin = 0 , sMin = 0, vMin = 137), (hMax = 179 , sMax = 42, vMax = 255) ====== White

def compute_hls_white_yellow_binary(hls_img):
    lower = np.array([7, 0, 160], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")
    mask = cv2.inRange(hls_img, lower, upper)

    lower1 = np.array([0, 0, 150], dtype="uint8")
    upper1 = np.array([168, 224, 255], dtype="uint8")
    mask1 = cv2.inRange(hls_img, lower1, upper1)
    kernel = np.ones((7, 7), np.uint8)
    img_hls_white_yellow_bin = cv2.bitwise_xor(mask, mask1)
    img_hls_white_yellow_bin = cv2.dilate(img_hls_white_yellow_bin, kernel, iterations=3)
    return img_hls_white_yellow_bin



cap = cv2.VideoCapture("test.mp4")
src = np.float32([(500, 476),
                  (272, 680),
                  (1262, 658),
                  (720, 460)])

dest = np.float32([(0, 0),
                   (0, 720),
                   (1270, 715),
                   (1270, 0)])
r, x = cap.read()
th2 = np.zeros_like(x)

while cap.isOpened():
    ret, frame = cap.read()
    kernel = 15
    if ret == True:
        frame = cv2.GaussianBlur(frame, (kernel, kernel), 1)
        LAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        HLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        (L, A, B) = cv2.split(LAB)
        (H, Lu, S) = cv2.split(HLS)
        # th1 = compute_hls_white_yellow_binary(HLS)


        M = cv2.getPerspectiveTransform(src, dest)
        M_inv = cv2.getPerspectiveTransform(dest, src)
        op = front_top(frame, M)
        x = compute_hls_white_yellow_binary(op)


        if (x.sum() < 20000000):
            th2 = x
        else:
            th2 = th2

        coords = np.column_stack(np.where(th2 > 0))

        op = curve(op, coords)
        op2 = cv2.bitwise_or(top_front(op, M_inv), frame)

        # d = np.hstack((cv2.resize(L, (200, 200)), cv2.resize(th2, (200, 200))))
        disp = np.hstack((cv2.resize(frame, (200, 200)), cv2.resize(op, (200, 200))))
        cv2.imshow("view", op2)
        cv2.imshow("op", disp)
        # cv2.imshow("op2", d)

        z = cv2.waitKey(1) & 0xFF
        if z == 27:
            break
    else:
        z = cv2.waitKey(1) & 0xFF
        if z == 27:
            break
cap.release()
cv2.destroyAllWindows()