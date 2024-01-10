from scipy import ndimage
from scipy.ndimage import convolve
import numpy as np

import imageio.v3
import sys

class cannyEdgeDetector:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.imgs = imgs
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return 
    
    
    def grayscale(self, img):
        """Convert an image to grayscale."""
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray
    
    
    def gaussian_kernel(self, size, sigma=1):
        """Get gaussian kernel with a specific size and sigma value."""
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    
    
    def gaussian_filter(self, img):
        """
        First step of canny-edge-detection.
        Convolve grayscale-image with a gaussian kernel.
        """
        return convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
    
    
    def sobel_filters(self, img):
        """
        Second step of canny-edge-detection.
        Gradient image and gradient angle are calculated.
        """
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        # Calculate gradient
        Ix = ndimage.convolve(img, Kx)
        Iy = ndimage.convolve(img, Ky)

        # Get gradient image
        G = np.hypot(Ix, Iy) 
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix) # Get gradient angle
        return (G, theta)
    

    def non_max_suppression(self, img, D):
        """
        Third step of canny-edge-detection.
        Makes sure all edges are only 1 pixel wide.
        """
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(0,M-1):
            for j in range(0,N-1):
                q = 0
                r = 0

                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    if(j < N-1):
                        q = img[i, j+1]
                    if(j > 0):
                        r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    if(i<M-1 and j > 0):
                        q = img[i+1, j-1]
                    if(i>0 and j < N-1):
                        r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    if(i < M-1):
                        q = img[i+1, j]
                    if(i > 0):
                        r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    if(i > 0 and j > 0):
                        q = img[i-1, j-1]
                    if(i < M-1 and j < N-1):
                        r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

        return Z


    def threshold(self, img):
        """
        Fourth step of canny-edge-detection.
        Group pixels with thresholds into different categories: No edge, weak edge, strong edge.
        """

        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold) # Strong pixels
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold)) # Weak pixels
        # All other pixels are not part of a edge

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)


    def hysteresis(self, img):
        """ 
        Fifth and final step of canny-edge-detection.
        Weak pixels adjacent to strong pixels are turned into strong pixels themselves.
        All other weak pixels are removed from the image.
        """
        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel
        
        # First loop: Find all weak pixels that can be turned to strong pixels
        i = -1
        while(i < M-1):       # Vertical coordinates
            i += 1
            j = -1
            while(j < N-1):   # Horizontal coordinates
                j += 1
                if (img[i,j] == weak):
                    # Booleans to check whether if there are neighbor pixels which are marked as strong
                    # Predefined as False
                    up_left = up = up_right = left = right = down_left = down = down_right = False
                    if(i > 0 and j > 0):
                        up_left = img[i-1, j-1] == strong
                    if(i > 0):
                        up = img[i-1, j] == strong
                    if(i > 0 and j < N-1):
                        up_right = img[i-1, j+1] == strong
                    if(j > 0):
                        left = img[i, j-1] == strong
                    if(j < N-1):
                        right = img[i, j+1] == strong
                    if(i < M-1 and j > 0):
                        down_left = img[i+1, j-1] == strong
                    if(i < M-1):
                        down = img[i+1, j] == strong
                    if(i < M-1 and j < N-1):
                        down_right = img[i+1, j+1] == strong
                        
                    # Turn current pixel strong if there is a strong pixel in neighborhood
                    if( up_left or up or up_right or
                      left or right or
                      down_left or down or down_right):
                        img[i, j] = strong
                        # Go back one row and one column
                        # to check for more weak pixels that can be turned into strong pixels
                        if (i > 0):
                            i -= 1
                        if (j > 0):
                            j -= 2   
                
        # Second loop: All remaining weak pixels are removed
        for i in range(0, M-1):
            for j in range(0, N-1):
                if (img[i,j] == weak):
                    img[i, j] = 0

        return img
    
    
    def detect(self):
        for img in self.imgs:    
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)
        return self.imgs_final


# If this file is executed itself, it will run the canny-edge-detection for an image (in grayscale)
# Parameters/options can be configured here
# Image is selected by file name. Allowed file names are: "input.png" and "input.jpg"
if __name__ == "__main__":
    ### Parameters for canny-edge-dection
    sigma = 1             # Determines how much the image should be blurred
    kernel_size = 5       # Size of the gaussian kernel, which will be convolved with the image. Value should be set higher if sigma is increased
    weak_pixel = 75       # Brightness of weak pixels in images
    strong_pixel = 255    # Brightness of strong pixels in images
    lowthreshold = 0.05   # Lower threshold for weak pixels
    highthreshold = 0.15  # Upper threshold for strong pixels
    
    output_image_for_each_step = True  # If set to true, an png-file is being created after each step. Otherwise, only the final result will be saved.

    
    canny = cannyEdgeDetector([], sigma, kernel_size, weak_pixel, strong_pixel, lowthreshold, highthreshold)
    
    # "Step 0": Read file
    try:
        img = imageio.v3.imread("./input.png")
    except FileNotFoundError:
        try:
            img = imageio.v3.imread("./input.jpg")
        except FileNotFoundError:
            print("Error: Input image could not be found in this directory")
            sys.exit(1)
    
    # Step 1: Create grayscale image and apply gaussian blur
    img_gray = canny.grayscale(img)
    img_filtered = canny.gaussian_filter(img_gray)
    if(output_image_for_each_step):
        imageio.v3.imwrite("./output_1_filtered.png", np.round(img_filtered.astype(np.uint8)))
    
    # Step 2: Calculate gradient and gradient angles
    img_grad, theta = canny.sobel_filters(img_filtered)
    if(output_image_for_each_step):
        imageio.v3.imwrite("./output_2_gradient.png", np.round(img_grad.astype(np.uint8)))
    
    # Step 3: Non-maximal suppression
    img_nms = canny.non_max_suppression(img_grad, theta)
    if(output_image_for_each_step):
        imageio.v3.imwrite("./output_3_nms.png", np.round(img_nms.astype(np.uint8)))
    
    # Step 4: Double Threshold
    img_thresh = canny.threshold(img_nms)
    if(output_image_for_each_step):
        imageio.v3.imwrite("./output_4_threshold.png", np.round(img_thresh.astype(np.uint8)))
        
    # Step 5: Hysteresis
    img_final = canny.hysteresis(img_thresh)
    imageio.v3.imwrite("./output_result.png", np.round(img_final.astype(np.uint8)))