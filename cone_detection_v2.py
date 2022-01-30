import cv2
import numpy as np

class cone_detect:
    def getCounters(img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cone_pixels = []
        centres = []
        for ind, contour in enumerate(contours):
            if cv2.contourArea(contour) > 100:
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centre = (cX, cY)
                centres.append(centre)
                mask = np.zeros_like(img)
                cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
                pts = np.where(mask == 255)
                cur_cone = []
                for i in range(len(pts[0])):
                    cur_cone.append([pts[0][i], pts[1][i]])
                cone_pixels.append([cur_cone])

        return centres, cone_pixels



    def getColour(img, color):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        low_orange = np.array([max(color - 20, 0), 0, 0])
        high_orange = np.array([min(color + 20, 255), 255, 255])
        x = cv2.inRange(img_hsv, low_orange, high_orange)
        x = cv2.bitwise_and(img, img, mask=x)
        return x

    def post_process(img):
        blur = cv2.GaussianBlur(img, (7, 7), 4)
        canny = cv2.Canny(blur, 75, 75)
        dilated = cv2.dilate(canny, (2, 2), iterations=1)
        blur = cv2.GaussianBlur(dilated, (3, 3), 1)
        return blur

    def get_pixels(self, img):
        try:
            original_image = cv2.imread(img)
        except:
            original_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        color_image = self.getColour(original_image, 2)
        post = self.post_process(color_image)
        centres, cones = self.getCounters(post)

        for ind, cone in enumerate(cones):
            for pixel in cone[0]:
                original_image[pixel[0], pixel[1]] = [255, 40, 0]

        return cones, centres
