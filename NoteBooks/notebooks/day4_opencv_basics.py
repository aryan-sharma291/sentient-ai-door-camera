import cv2
import numpy as np


img = np.zeros((512, 512, 3), np.uint8)

img = cv2.rectangle(img, (10, 10), (307, 307), (153,217,191), 5)
img = cv2.circle(img, (100,100), 50, (153,217,191), 1)
img = cv2.line(img, (150,150), (250,250), (153,217,191), 1)
img = cv2.putText(img,"Hi I did this",(180,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2 )
cv2.imshow("image", img)

if cv2.waitKey(0) == ord('s'):
    save = cv2.imwrite("hi.png", img)