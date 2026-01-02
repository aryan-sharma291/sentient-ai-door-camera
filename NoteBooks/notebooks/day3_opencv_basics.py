import cv2

print("OpenCV:", cv2.__version__)


image = cv2.imread("picture.jpeg")
cv2.imshow("image", image)
k = cv2.waitKey(0)
img = cv2.imread("picture.jpeg", cv2.IMREAD_GRAYSCALE)
if k == ord('q'):
    cv2.imwrite("picture.jpeg", image)

cv2.imshow("image 2",img)
k2 = cv2.waitKey(0)

if k2 == ord('a'):
    cv2.destroyAllWindows()