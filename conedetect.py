import cv2
import numpy as np


def getCounters(img, img2):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cone_pixels = []
    for ind, contour in enumerate(contours):
        if cv2.contourArea(contour)> 100:
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [contour], -1, 255,thickness=-1)
            pts = np.where(mask == 255)
            cur_cone = []
            for i in range(len(pts[0])):
                cur_cone.append([pts[0][i], pts[1][i]])
            cone_pixels.append([cur_cone])

    return mask, cone_pixels

def getColour(img, color):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_orange = np.array([max(color-20, 0), 0, 0])
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



original_image = cv2.imread(r"C:\Main Folder\Unity\stereo\Assets\screenshots\screen_1960x1080_2021-12-29_23-29-13.png")
#cv2.imshow("original",original_image)
#cv2.waitKey(0)
#original_image = cv2.resize(original_image, (int(original_image.shape[0] * 50/100),(int(original_image.shape[1] * 50/100))), interpolation= cv2.INTER_AREA )

color_image = getColour(original_image, 2)
#cv2.imshow("just orange",color_image)

post = post_process(color_image)
#cv2.imshow("after processing", post)

coutners, cones = getCounters(post, original_image.copy())
#cv2.imshow("coutners", coutners)
print(cones)
print(cones[0])

for ind,cone in enumerate(cones):
    for pixel in cone[0]:
        original_image[pixel[0],pixel[1]] = [255,40,0]
    print(ind)
cv2.imshow("afd", original_image)

"""
empty_image = np.zeros((original_image.shape[0],original_image.shape[1],3), dtype=np.uint8)

firstImage = getCounters(post_process(getColour(original_image,2)), original_image.copy())
secondImage = getCounters(post_process(original_image), original_image.copy())


finalImage = cv2.bitwise_and(firstImage,secondImage)

window = np.concatenate((firstImage, secondImage, finalImage), axis=1)
cv2.imshow("img", window)
#cv2.imshow("canny", canny)

#cv2.imshow("edges", drawImage)
"""
cv2.waitKey(0)