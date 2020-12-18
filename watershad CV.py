import imutils
import cv2
import time

start_time = time.time()

image = cv2.imread("test.jpg")
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
print("Затраченное время на выполнение алгоритма инструментами OpenCV: %s seconds ---" % (time.time() - start_time))
cv2.imshow("Input", image)