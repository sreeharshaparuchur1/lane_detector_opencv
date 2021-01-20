import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
def diff_of_gaussians(img):

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur_img_grey = cv2.GaussianBlur(grey_img, (9,9), 0)
    blur_img_colour = cv2.GaussianBlur(img, (9,9), 0)

    #plt.figure(figsize = (20,2))
    #plt.imshow(blur_img_grey, cmap = 'gray')
    #plt.show()
    #plt.imshow(blur_img_colour)
    #plt.show()

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows = 2, ncols = 2)
    edges_grey = cv2.Canny(grey_img,100,200)
    edges = cv2.Canny(img, 100, 200)
    #plt.subplot(411)
    ax1.imshow(edges_grey, cmap = 'gray')
    #plt.imshow(edges_grey, cmap = 'gray')
    #plt.show()
    #plt.subplot(412)
    ax2.imshow(edges);
    #plt.imshow(edges)
    #plt.subplot(421)
    ax3.imshow(canny(grey_img), cmap = 'gray')
    #plt.show()
    #plt.subplot(422)
    ax4.imshow(canny(img))
    #plt.show()

    plt.show()
    #plt.imshow(blur_img_grey - grey_img, cmap = 'gray')
    #plt.show()
    #plt.imshow(blur_img_colour - img)
    #plt.show()

    return
'''

def canny(img):
    # changes in intensity are to be captured.
    # Canny and the Sobel operator work no greyscale images
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian Blurring to reduce noise, removes high-frequncy components in the image
    # High frequency due to high ISO of the camera, contours that aren't really edges.
    # https://www.youtube.com/watch?v=uihBwtPIBxM
    blurred_img = cv2.GaussianBlur(grey_img, (9,9), 0)

    # Canny Edge Detector, identifying any sharp changes in intesity, Uses edge-gradients
    # the strongest gradents are then traced
    # https://www.youtube.com/watch?v=sRFM5IEqR2w
    canny_filtered = cv2.Canny(blurred_img, 30, 150)
    return canny_filtered

def region_of_interest(img):
    height, width = img.shape
    height -= 60
    width -= 10
    #Reducing the image size to "focus" more on the center of the frame (region of interest)
    #These dimensions are later used in the generation of the mask
    #The reduction in height enables us to ignore the part of the image corresponding to the dashboard.

    #Coordinates marking our "region of interest"
    #The top-left of the image is (0,0)
    Polygons = np.array([
        [(width, height),(50, height), (int((3/8) * width), int((3/4) * height)),(int((5/8) * width), int((3/4) * height))]
        ])
    #(width, height),(50,height) removes what's visible of the dash of the car.
    mask = np.zeros_like(img)
    # filling mask
    cv2.fillConvexPoly(img = mask, points = Polygons, color = 255, lineType = cv2.LINE_AA)
    # Uncomment "return mask" to see the "region of interest" marked in white
    mask_img = cv2.bitwise_and(img, mask)
    # mask_img now has the detected edges in our region of interest.
    #return mask
    return mask_img

def dispay_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_img, (x1,y1), (x2,y2), (0,255,0), 30)
    return line_img

def get_cords(img, line_slope_int):
    slope, intercept = line_slope_int
    y1 = img.shape[0]
    #Line starts from the bottom left
    y2 = int(y1 * (4/5))
    # The line goes 1 fifth of the way up
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    #from y = mx + c
    #print(img.shape)
    height, width, _ = img.shape
    if x1 > width or x1 < 0 or x2 > width or x2 < 0 or y1 > height or y1 < 0 or y2 > height or y2 < 0:
        return np.array([0,0,0,0])
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    #if lines is None:
    #    return (np.array([0,0,0,0]), np.array([0,0,0,0]))
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope, intercept = np.polyfit((x1,x2), (y1,y2), 1)
        #Linear least squares :) (not exactly but it's easy to think of it like this)
        print(slope, intercept)
        #left lines have a positive slope.
        if slope >= 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if left_fit:
        left_fit_avg = np.average(left_fit, axis = 0)
        left_line = get_cords(img, left_fit_avg)
    else:
        left_line = np.array([0,0,0,0])
    if right_fit:
        right_fit_avg = np.average(right_fit, axis = 0)
        right_line = get_cords(img, right_fit_avg)
    else:
        right_line = np.array([0,0,0,0])

    return np.array([left_line, right_line])

if __name__ == "__main__":
    cap = cv2.VideoCapture("./../Downloads/detect_lanes_from.mp4")
    lines = np.asarray((np.array([0,0,0,0]), np.array([0,0,0,0])))
    estimate = lines
    while (cap.isOpened()):
        _, frame = cap.read()
        canny_img = canny(frame)
        masked_img = region_of_interest(canny_img)
        estimate = lines
        #print(len(estimate), len(lines))
        # Finding straight lines and therefore the lane lines --> Hough transform
        lines = cv2.HoughLinesP(masked_img, 1, (np.pi / 180), 100, np.array([]), minLineLength = 10, maxLineGap = 500)
        #print(estimate.shape, lines.shape)
        if lines is None:
            lines = estimate
        # https://www.youtube.com/watch?v=4zHbI-fFIlI watch at 1.5x lol
        avg_lines = average_slope_intercept(frame, lines)
        #print(avg_lines)
        line_img = dispay_lines(frame, avg_lines)
        img_frame = cv2.addWeighted(frame, 1, line_img, 0.8, 0)

        cv2.imshow("colour_camera_frame", img_frame)
        cv2.imshow("contoured", masked_img)
        if cv2.waitKey(2) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
