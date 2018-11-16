import cv2
import numpy as np

k = 50

image = cv2.imread('prudential.jpg')

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('original_image.png', grayImage)

print("\nThe rank of the original image matrix is: %d\n" % np.linalg.matrix_rank(grayImage))

# print(grayImage)
# print(grayImage.shape)
# cv2.imshow('Original image',image[200:1024,:])
# cv2.imshow('Gray image', grayImage[200:1024,:])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

u, s, v = np.linalg.svd(grayImage)

# print(u.shape)
# print(s.shape)
# print(v.shape)
# print(s)
# print(np.allclose(grayImage, np.dot(u[:, :768] * s, v)))

u_k = u[:,0:k]
s_k = s[0:k]
v_k = v[0:k,:]

# print(u.shape)
# print(s.shape)
# print(v.shape)

newImage = np.dot(u_k[:, :k] * s_k, v_k)

print("The rank of the new approximated image matrix is: %d\n" % np.linalg.matrix_rank(newImage))

newImage = np.array(newImage ,dtype=np.uint8)

new_filename = 'rank_' + str(k) + '_approx_image.png'
cv2.imwrite(new_filename, newImage)

# print(newImage)
# cv2.imshow('New Image', newImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()