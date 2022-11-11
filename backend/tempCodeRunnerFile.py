eigfaces = np.array(get_eigenfaces(eigenvec,40))


# a = eigfaces[0].reshape(256,256)
# a = np.interp(a, (a.min(), a.max()), (0, 256))
# img = (a).reshape(256,256).astype('uint8')

# grayImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# cv2.imshow('image',grayImage)
# cv2.waitKey(0)