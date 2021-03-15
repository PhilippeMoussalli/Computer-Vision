import numpy as np
import matplotlib
import cv2
import imutils
matplotlib.use("TkAgg")
img = cv2.imread('edgess.jpg')
nb_objects = 1
from func import *

class frame:

    def __init__(self,name,image):
        self.name = name
        self.image = image
        self.rows= image.shape[0]
        self.cols = image.shape[1]

    def show_image(self,img):
        cv2.imshow(self.name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def make_rgb_masks(self,color='W'):

        rgb_matrix = np.ones((self.rows,self.cols), np.uint8)*255
        zero_matrix = np.zeros((self.rows,self.cols), np.uint8)

        if color =='W':
            color_matrix =  cv2.merge((rgb_matrix,rgb_matrix,rgb_matrix))
        elif color == 'B':
            color_matrix = cv2.merge((rgb_matrix, zero_matrix, zero_matrix))
        elif color =='G':
            color_matrix = cv2.merge((zero_matrix, rgb_matrix, zero_matrix))
        else:
            color_matrix = cv2.merge((zero_matrix, zero_matrix, rgb_matrix))

        return color_matrix

    def convert_2_gray(self,dim=None,display=False):

        grey_1D = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        conv_image = grey_1D

        if dim==3:
            grey_3D = cv2.cvtColor(grey_1D , cv2.COLOR_GRAY2BGR)
            conv_image = grey_3D

        if display:
            self.show_image(conv_image)

        return conv_image

    def gaussian_blur(self,kernel_width,kernel_height,sd,display=False):

        blured_image_gauss = cv2.GaussianBlur(self.image,(kernel_width,kernel_height),sd)

        if display:
            self.show_image(blured_image_gauss)

        return blured_image_gauss

    def blur_bilateral(self,d,sigmaColor,sigmaSpace,display=False):

        blured_image_bi = cv2.bilateralFilter(self.image,d,sigmaColor,sigmaSpace)

        if display:
            self.show_image(blured_image_bi)

        return blured_image_bi

    def sobel_edge(self,sobel_kernel_size,display=False):

        image_display = self.image
        image_sobel = self.gaussian_blur(3, 3, 10)

        sobel_h = cv2.Sobel(image_sobel, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
        color_matrix = self.make_rgb_masks('R')
        image_display [sobel_h>300] = color_matrix[sobel_h>300]

        sobel_v = cv2.Sobel(image_sobel, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
        color_matrix = self.make_rgb_masks('G')
        image_display [sobel_v>300] = color_matrix[sobel_v>300]

        if display:
            self.show_image(image_display )

        return image_display

    def color_object_detection (self,hsv_lower_range,hsv_upper_range, fillholes = False
                                ,display=False,morphological = False,color='W'):

        # Convert image to hsv
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # Obtain image mask by detection for thresholded value
        mask = cv2.inRange(hsv, hsv_lower_range, hsv_upper_range)

        # Optional: morphological improvements on images
        if morphological:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((6, 6), np.uint8))

        imask = mask > 0


        # Create a blank matrix with same size as original image
        object = np.zeros_like(self.image, np.uint8)
        # Fill blank matrix with image values that correspond to the object
        object[imask] = self.image[imask]


        # Improvement: filling holes and setting background values to zero
        if fillholes:

            # Find countours from thresholded value
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Get largest contours that correspond to the object
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)
            # Get bounding box used to exclude other values outside it to zero
            x, y, w, h = cv2.boundingRect(big_contour)
            imask_box = np.zeros((self.rows, self.cols))
            imask_box[y:y + h, x:x + w] = 1
            # draw filled contour on black background
            cv2.drawContours(mask, [big_contour], 0, (255, 255, 255), -1)
            # Matrix is elements of the detected object within the bounding box
            imask = np.logical_and(mask > 0, imask_box > 0)
            object = np.zeros_like(self.image, np.uint8)
            object[imask] = self.image[imask]

        if display:
            self.show_image(object)

        color_matrix = self.make_rgb_masks(color)
        mask_matrix = np.zeros_like(self.image, np.uint8)
        mask_matrix[imask] = color_matrix [imask]
        return mask_matrix

        #return object

    def circle_hough_transform(self,median_filter_kernel_size = 19,dp=1,scale = 8,param1 = 10,param2 = 60,
                               minRadius = 30, maxRadius = 150,draw_on_grey=True,display=False,output_all=True,
                               additional_box = None,ssd_previous=None):

        """
        4) Hough transform

            param1: threshold value of Canny edge detection --> lower value will result in more
            weak edges that are detected

            param2: param2 is the threshold value for the final selection of elements from the Accumulator Matrix,
            in other words how many edge points need to to be found to declare that a circle was found
            min-max radius --> dont set them too far apart unless you want all possible circles that might be
            found in that range
        """
        # Noise Removal

        img_hough_blur_1D = cv2.medianBlur(self.convert_2_gray(dim=1),median_filter_kernel_size)
        img_hough_blur_3D = cv2.cvtColor(img_hough_blur_1D, cv2.COLOR_GRAY2BGR)

        # Hough transform
        circles = cv2.HoughCircles(img_hough_blur_1D, cv2.HOUGH_GRADIENT, dp=dp, minDist=self.rows/scale,
                                   param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        # Specify whether to show th drawing on a grey image or original image
        if draw_on_grey:
            image_display = self.convert_2_gray(dim=3)
        else:
            image_display = self.image

        if circles is not None:
            circles = np.uint16(np.around(circles))
            if output_all:
                circles_plot = circles[0][0:]
            else:
                circles_plot = circles[0][0:1]

            for i in circles_plot:

                # Rectangle Center coordinate
                (x_0, y_0, r_0) = (i[0], i[1], i[2])
                coords = (x_0, y_0, r_0)
                # Draw outer circle
                cv2.circle(image_display, (x_0, y_0), r_0, (0, 255, 0), 2)
                # Draw inner circle
                cv2.circle(image_display, (x_0, y_0), 2, (0, 0, 255), 3)
                # Draw rectangle
                cv2.rectangle(image_display, (x_0 - r_0, y_0 - r_0), (x_0 + r_0, y_0 + r_0), (255, 0, 0), 2)
                ssd = None
                # Draw rectangle from first half for grey level transformation
                if additional_box is not None:
                    (x_fh, y_fh, r_fh) = (additional_box[0], additional_box[1], additional_box[2])
                    cv2.rectangle(image_display, (x_fh- r_fh, y_fh- r_fh),(x_fh+ r_fh,y_fh + r_fh), (0, 255, 0), 2)
                    ssd = np.sqrt((x_fh - x_0) ** 2 + (y_fh - y_0) ** 2)
                    if math.isnan(ssd):
                        ssd = ssd_previous

                    # Scale values such that an ssd of the width to have zero intensity
                    if ssd is None:
                        ssd = 1
                    scale =  ((11*ssd)/self.cols)+1
                    print(scale)
                    image_display = np.uint8(image_display//scale)


        if display:
            self.show_image(image_display)

        return image_display,coords,ssd


    @staticmethod
    def multiscale_template_matching(original_frame,template,first_box=True,coords=[],center=[]):

        # Calculate edges of both original images and template
        if first_box == True:
             image = original_frame.image
        else:
            image = original_frame.convert_2_gray(dim=3)

        original_frame_grey = original_frame.convert_2_gray(dim=1)
        original_frame_edge = cv2.Canny(original_frame_grey , 50, 200)

        template_grey = template.convert_2_gray(dim=1)
        template_edge = cv2.Canny(template_grey,50, 200)

        scales = np.linspace(0.2, 2, 20)[::-1]
        found = None

        for scale in scales:
            # resize the image according to the scale
            resized = imutils.resize(original_frame_edge, width=int(original_frame.cols * scale))
            # Keep track of ratio of resizing
            r = original_frame.cols / float(resized.shape[1])

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < template.rows or resized.shape[1] < template.cols:
                break

            # Compute the canny edge for the resized image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template_edge, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # Unpack the values
        (_, maxLoc, r) = found
        (coord_0_X, coord_0_Y) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (coord_1_X, coord_1_Y) = (int((maxLoc[0] + template.cols) * r), int((maxLoc[1] + template.rows) * r))

        # draw a bounding box around the detected result and display the image
        if first_box:
            cv2.rectangle(image, (coord_0_X, coord_0_Y), (coord_1_X, coord_1_Y), (0, 0, 255), 2)
            coords = [coord_0_X, coord_0_Y, coord_1_X, coord_1_Y]
            center_coord = ((coord_1_X - coord_0_X), (coord_1_Y - coord_0_Y))
        else:
            cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
            center_coord = ((coord_1_X - coord_0_X), (coord_1_Y - coord_0_Y))
            cv2.rectangle(image, (coord_0_X, coord_0_Y), (coord_1_X, coord_1_Y), (0, 255, 0), 2)
            print("New_",center_coord)
            print("old_",center)
            ssd =  np.sqrt((center_coord[0]-center[0])**2 + (center_coord[1]-center[1])**2)
            print(ssd)



        return image, center_coord, coords

    @staticmethod
    def feature_matching(original_frame,template):

        img = original_frame.image
        temp = template.image

        # Initialize the scale invariant feature transform object
        sift = cv2.xfeatures2d.SIFT_create()

        # Compute the key points and their descriptors. Descriptors are independent of key point position, robust
        # against image transformation, and scale independent

        kp_img,desc_img = sift.detectAndCompute(img,None)
        kp_temp, desc_temp= sift.detectAndCompute(temp, None)

        # Create brute force matcher
        bf = cv2.BFMatcher()

        # Compute the best two matches for each descriptor

        matches = bf.knnMatch(desc_img,desc_temp,k=2)

        # Apply ratio test (keep descriptors that are close to each other)

        good_matches = []
        for match1,match2 in matches:
            if match1.distance < 0.6* match2.distance:
                good_matches.append([match1])


        sift_matches = cv2.drawMatchesKnn(img,kp_img,temp,kp_temp,good_matches,None,flags=2)

        return sift_matches



