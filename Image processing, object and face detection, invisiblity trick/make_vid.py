from main2 import frame

import cv2
import time

fps = 30
video_name = "part1.mp4"
cap = cv2.VideoCapture(video_name)
frame_cnt = 0

# Color space of object to grab
yellow_lower_limit = (25, 40, 40)
yellow_upper_limit = (70, 255, 255)
template_ball_1 = frame('ball_template_1',cv2.imread('template_ball_7.jpg'))

frameSize = (1280,720)
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)
make_video = False

def write_subtitles(img,text,rows,cols):
    cv2.putText(img,text, (int(rows//2),int(cols/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

if cap.isOpened() == False:
    print("ERROR FILE NOT FOUND OR WRONG CODEC USED !")

while cap.isOpened():

    second_passed = frame_cnt/fps
    ret,frame_cap = cap.read()

    if ret == True:

        #Play at normal speed
        #time.sleep (1/fps)

        # Create frame object
        frame_display = frame('frame_'+str(frame_cnt),frame_cap)
        rows = frame_display.rows
        cols= frame_display.cols
        ## Here we will do the image processing for different seconds

        """
        1) Switch between color and greyscale a few times (4s)
        """
        if 0<=second_passed<=1:
            frame_display=frame_display.convert_2_gray(dim=3)
        if 1<second_passed<=2:
            frame_display = frame_display.image

        if 2 < second_passed <= 3:
            frame_display=frame_display.convert_2_gray(dim=3)

        if 3 < second_passed <= 4:
            frame_display = frame_display.image

        """
        2) Smoothing or blurring with gaussian and bilateral filters
        """

        if 4 < second_passed <= 6:
            frame_display = frame_display.gaussian_blur(3,3,2)
            write_subtitles(frame_display,'Gaussian Blur with 3x3 kernel and sigma = 2', rows, cols)

        if 6 < second_passed <= 8:
            frame_display = frame_display.gaussian_blur(15,15,2)
            write_subtitles(frame_display, 'Gaussian Blur with 15x15 kernel and sigma = 2 ', rows, cols)

        if 8 < second_passed <= 10:
            frame_display = frame_display.gaussian_blur(15,15,10)
            write_subtitles(frame_display, 'Gaussian Blur with 15x15 kernel and sigma = 10 ', rows, cols)


        if 10 < second_passed <= 12:
            frame_display = frame_display.blur_bilateral(2,75,75)
            write_subtitles(frame_display, 'Bilateral filter (edge preservant) with d = 2 ', rows, cols)

        if 12 < second_passed <= 14:
            frame_display = frame_display.blur_bilateral(10,75,75)
            write_subtitles(frame_display, 'Bilateral filter (edge preservant) with d = 10 ', rows, cols)

        """
        3) Grabbing the object
        """

        if 14 < second_passed <= 17:
            frame_display = frame_display.color_object_detection(yellow_lower_limit, yellow_upper_limit, fillholes=False,
                                                 morphological=False,color='W')
            write_subtitles(frame_display, 'Simple object grabbing with thresholding ', rows, cols)

        if 17 < second_passed <= 20:
            frame_display = frame_display.color_object_detection(yellow_lower_limit, yellow_upper_limit, fillholes=False,
                                                 morphological=True,color='G')
            write_subtitles(frame_display, 'Improvement with opening (dilation followed by erosion)', rows, cols)


        if 20 < second_passed <= 24:
            frame_display = frame_display.color_object_detection(yellow_lower_limit, yellow_upper_limit, fillholes=True,
                                                 morphological=True,color='B')
            write_subtitles(frame_display, 'Improvement by filling thresholded values withing bounding box of the largest'
                                           ' contour', rows, cols)

        """
        4) Sobel Edges
        """

        if 24 < second_passed <= 26:
            frame_display = frame_display.sobel_edge(7)
            write_subtitles(frame_display, 'Sobel edges, horizontal (R), Vertical (G), kernel = 7', rows, cols)

        if 26 < second_passed <= 29:
            frame_display = frame_display.sobel_edge(5)
            write_subtitles(frame_display, 'Sobel edges, horizontal (R), Vertical (G), kernel = 5', rows, cols)

        if 29 < second_passed <= 31:
            frame_display = frame_display.sobel_edge(3)
            write_subtitles(frame_display, 'Sobel edges, horizontal (R), Vertical (G), kernel = 3', rows, cols)

        """
        5) Hough circles 
        """

        if 31 < second_passed <= 34:
            frame_display,_,_  = frame_display.circle_hough_transform(draw_on_grey=False,scale=16, dp=0.5, param1=7)
            write_subtitles(frame_display, 'Lots of overlap: Increase the distance between detections (mindist) ', rows, cols)

        if 34 < second_passed <= 37:
            frame_display,_,_  = frame_display.circle_hough_transform(draw_on_grey=False,scale=4, dp=0.5, param1=7)
            write_subtitles(frame_display, 'hmm.. Still very noisy, maybe we can increase the threshold for the Canny edge', rows, cols)

        if 37 < second_passed <= 41:
            frame_display,_,_  = frame_display.circle_hough_transform(draw_on_grey=False,scale = 4,dp=0.5, param1=9)
            write_subtitles(frame_display, 'This seems better (param1 increased), now we are detecting stronger edges',
                        rows, cols)

        if 41 < second_passed <= 45:
            frame_display,_,_  = frame_display.circle_hough_transform(draw_on_grey=False,scale =4, dp=0.5, param1=11)
            write_subtitles(frame_display, 'If we keep increasing this parameter we will only detect few circles',
                        rows, cols)

        """
        5) Box probability and inverse SSD
        """

        if 45<second_passed<=48:
            frame_display, coords, ssd = frame_display.circle_hough_transform(draw_on_grey=False, dp=1, scale=1,
                                                                              param1=10
                                                                              , minRadius=20, maxRadius=220,
                                                                              output_all=False)
            write_subtitles(frame_display, 'Let us calculate the probability that the ball is within the bounding box',
                        rows, cols)

        if 48  < second_passed <= 57:
            frame_display,_,ssd= frame_display.circle_hough_transform(draw_on_grey=True, dp=1,scale=1, param1=10
                                                                 ,minRadius=20,maxRadius=220,output_all=False,
                                                                   additional_box=coords,ssd_previous=ssd)

            write_subtitles(frame_display, 'Decrease of brightness assosicated with SSD between the ball and the bounding box',
                        rows, cols)
        """
        6) Feature matching 
        """

        if 57  <second_passed:
            frame_display=frame_display.feature_matching(frame_display, template_ball_1)
            frame_display = cv2.resize(frame_display, dsize=frameSize, interpolation=cv2.INTER_CUBIC)
            write_subtitles(frame_display, 'Feature detection through scale invariant feature transform (SIFT)',
                        rows, cols)
            if second_passed == 72:
                out.release()
        # Write to video writer
        if make_video:
            out.write(frame_display)
        cv2.imshow("frame",frame_display)
        frame_cnt+=1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    else:
        break
out.release()
cap.release()
cv2.destroyAllWindows()