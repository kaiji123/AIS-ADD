from sklearn.cluster import KMeans
import cv2
import numpy as np
# Load the image and convert it to grayscale
image = cv2.imread("C:\\Users\\Kai Ji\\Desktop\\Maskformer\\MaskFormer\\datasets\\far\\test\\images\\90.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reshape the image into a feature vector
features = gray.reshape(image.shape[0] * image.shape[1], 1)

# Use K-Means clustering to segment the image into "stuff" and "things"
kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
labels = kmeans.predict(features)

# Reshape the labels back into the original image shape
labels = labels.reshape(image.shape[0], image.shape[1])

# Create a mask for the "stuff" regions
mask = np.where(labels == 0, 255, 0).astype(np.uint8)

# Apply the mask to the image to highlight the "stuff" regions
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Show the original image and the masked image
cv2.imshow("Original Image", image)
cv2.imshow("Panoptic Segmentation", masked_image)
cv2.waitKey(0)