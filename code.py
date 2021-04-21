import numpy as np
from matplotlib import pyplot as plt
import argparse
import cv2
import sys


def nothing(x):
    pass

def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


cap = cv2.VideoCapture('input1.mp4',0)
fps = int(round(cap.get(5)))
#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
frame_width = 640
frame_height = 360
fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow("TrackbarsHSV")
cv2.createTrackbar("L - H", "TrackbarsHSV", 5, 179, nothing)
cv2.createTrackbar("L - S", "TrackbarsHSV", 180, 255, nothing)
cv2.createTrackbar("L - V", "TrackbarsHSV", 0, 255, nothing)
cv2.createTrackbar("U - H", "TrackbarsHSV", 25, 179, nothing)
cv2.createTrackbar("U - S", "TrackbarsHSV", 255, 255, nothing)
cv2.createTrackbar("U - V", "TrackbarsHSV", 255, 255, nothing)

cv2.namedWindow("TrackbarsRGB")
cv2.createTrackbar("L - R", "TrackbarsRGB", 37, 255, nothing)
cv2.createTrackbar("L - G", "TrackbarsRGB", 28, 255, nothing)
cv2.createTrackbar("L - B", "TrackbarsRGB", 4, 255, nothing)
cv2.createTrackbar("U - R", "TrackbarsRGB", 255, 255, nothing)
cv2.createTrackbar("U - G", "TrackbarsRGB", 137, 255, nothing)
cv2.createTrackbar("U - B", "TrackbarsRGB", 35, 255, nothing)


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame,(frame_width,frame_height),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        if cv2.waitKey(28) & 0xFF == ord('q'):
            break
        if between(cap, 40000, 60000):
            break
        
        ### 0-4 Switch between color and grayscale
        if between(cap, 0, 4000):
            ## 0-1 Color
            ## 1-2 Grayscale
            ## 2-3 Color
            ## 3-4 Grayscale
            if between(cap, 0, 1000) or between(cap, 2000, 3000):
                cv2.putText(frame, "Color", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.putText(frame, "Grayscale", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
        
        ### 4-12 Smoothing and Bluring
        if between(cap, 4000, 12000):
            ## 4-6 Gaussian Blur 5x5
            if between(cap, 4000, 6000):
                frame = cv2.GaussianBlur(frame,(5,5),0)
                cv2.putText(frame, "Gaussian Blur 5x5", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ## 8-10 Gaussian Blur 15x15
            if between(cap, 8000, 10000):
                frame = cv2.GaussianBlur(frame,(15,15),0)
                cv2.putText(frame, "Gaussian Blur 15x15", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ## 6-8 Bilateral Filtering 5x5
            if between(cap, 6000, 8000):
                frame = cv2.bilateralFilter(frame,5,5,75)
                cv2.putText(frame, "Bilateral Filtering 5x5", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ## 10-12 Bilateral Filtering 15x15
            if between(cap, 10000, 12000):                
                frame = cv2.bilateralFilter(frame,15,15,75)
                cv2.putText(frame, "Bilateral Filtering 15x15", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
        
        ### 12-20 Identify object in RGB and in HSV
        if between(cap, 12000, 20000):
            ### 12-14 RGB
            if between(cap, 12000, 14000):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                l_r = cv2.getTrackbarPos("L - R", "TrackbarsRGB")
                l_g = cv2.getTrackbarPos("L - G", "TrackbarsRGB")
                l_b = cv2.getTrackbarPos("L - B", "TrackbarsRGB")
                u_r = cv2.getTrackbarPos("U - R", "TrackbarsRGB")
                u_g = cv2.getTrackbarPos("U - G", "TrackbarsRGB")
                u_b = cv2.getTrackbarPos("U - B", "TrackbarsRGB")
                
                
                lower_blue = np.array([l_r, l_g, l_b])
                upper_blue = np.array([u_r, u_g, u_b])
                frame = cv2.bilateralFilter(rgb,15,15,75)
                mask = cv2.inRange(frame, lower_blue, upper_blue)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                cv2.putText(frame, "RGB Color", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ### 14-16 HSV
            if between(cap, 14000, 16000):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                l_h = cv2.getTrackbarPos("L - H", "TrackbarsHSV")
                l_s = cv2.getTrackbarPos("L - S", "TrackbarsHSV")
                l_v = cv2.getTrackbarPos("L - V", "TrackbarsHSV")
                u_h = cv2.getTrackbarPos("U - H", "TrackbarsHSV")
                u_s = cv2.getTrackbarPos("U - S", "TrackbarsHSV")
                u_v = cv2.getTrackbarPos("U - V", "TrackbarsHSV")

                lower_blue = np.array([l_h, l_s, l_v])
                upper_blue = np.array([u_h, u_s, u_v])
                #frame = cv2.GaussianBlur(frame,(5,5),0)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                cv2.putText(frame, "HSV Color", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ### 16-18 Dilation
            if between(cap, 16000, 18000):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                l_h = cv2.getTrackbarPos("L - H", "TrackbarsHSV")
                l_s = cv2.getTrackbarPos("L - S", "TrackbarsHSV")
                l_v = cv2.getTrackbarPos("L - V", "TrackbarsHSV")
                u_h = cv2.getTrackbarPos("U - H", "TrackbarsHSV")
                u_s = cv2.getTrackbarPos("U - S", "TrackbarsHSV")
                u_v = cv2.getTrackbarPos("U - V", "TrackbarsHSV")
                
                lower_blue = np.array([l_h, l_s, l_v])
                upper_blue = np.array([u_h, u_s, u_v])
                #frame = cv2.GaussianBlur(frame,(5,5),0)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                #frame = cv2.bitwise_and(frame, frame, mask=mask)
                
                kernel = np.ones((5,5), np.uint8)
                dilation = cv2.dilate(mask, kernel, iterations = 1)
                #frame = dilation
                #frame = cv2.bitwise_and(frame, frame, mask=mask)
                frame = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
                cv2.putText(frame, "Dilation", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ### 18-20 Opening
            if between(cap, 18000, 20000):
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                l_h = cv2.getTrackbarPos("L - H", "TrackbarsHSV")
                l_s = cv2.getTrackbarPos("L - S", "TrackbarsHSV")
                l_v = cv2.getTrackbarPos("L - V", "TrackbarsHSV")
                u_h = cv2.getTrackbarPos("U - H", "TrackbarsHSV")
                u_s = cv2.getTrackbarPos("U - S", "TrackbarsHSV")
                u_v = cv2.getTrackbarPos("U - V", "TrackbarsHSV")

                lower_blue = np.array([l_h, l_s, l_v])
                upper_blue = np.array([u_h, u_s, u_v])
                #frame = cv2.GaussianBlur(frame,(5,5),0)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                #frame = cv2.bitwise_and(frame, frame, mask=mask)
                kernel = np.ones((5,5), np.uint8)
                opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
                frame = opening
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                cv2.putText(frame, "Opening", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
        ### 20-26 Detect Edges
        if between(cap, 20000, 26000):
            ### 20-22 
            if between(cap, 20000, 22000):
                frame = cv2.GaussianBlur(frame,(5,5),0)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5,scale=2,delta=0,borderType=cv2.BORDER_DEFAULT)  
                grad_y = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5,scale=2,delta=0,borderType=cv2.BORDER_DEFAULT)  
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                cv2.putText(frame, "ksize=5, scale=2", (20,frame_height-20) , font, 0.6, (255,255,255) , 2, cv2.LINE_4)
            ### 22-24 
            if between(cap, 22000, 24000):
                frame = cv2.GaussianBlur(frame,(5,5),0)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)  
                grad_y = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)  
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                cv2.putText(frame, "ksize=5, scale=1", (20,frame_height-20) , font, 0.6, (255,255,255) , 2, cv2.LINE_4)
            ### 24-26 
            if between(cap, 24000, 26000):
                frame = cv2.GaussianBlur(frame,(3,3),0)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)  
                grad_y = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)  
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                cv2.putText(frame, "ksize=3, scale=1", (20,frame_height-20) , font, 0.6, (255,255,255) , 2, cv2.LINE_4)                
            cv2.putText(frame, "Sobel Edge Detection", (20,frame_height-50) , font, 0.6, (255,255,255) , 2, cv2.LINE_4)
         
        ### 26-35 Detect Shapes
        if between(cap, 26000, 35000):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            cv2.putText(frame, "Circle Detection - Hough Transform", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ## 26-29
            if between(cap, 26000, 29000):
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=15, minRadius=0, maxRadius=0)
                cv2.putText(frame, "param1=100, param2=15, minRadius=0, maxRadius=100", (20,frame_height-20) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ## 29-32
            if between(cap, 29000, 32000):
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=20, minRadius=0, maxRadius=100)
                cv2.putText(frame, "param1=100, param2=20, minRadius=0, maxRadius=100", (20,frame_height-20) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
            ## 32-35
            if between(cap, 32000, 35000):
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 40, param1=150, param2=20, minRadius=20, maxRadius=40)
                cv2.putText(frame, "param1=150, param2=20, minRadius=20, maxRadius=40", (20,frame_height-20) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
                
            # Draw circles
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x,y,r) in circles:
                    cv2.circle(frame, (x,y), r, (36,255,12), 3)
            
        ### 35-40 Detect Object
        if between(cap, 35000, 40000):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            template = cv2.imread('template.jpg',0)
            template = cv2.resize(template, (69, 69), interpolation=cv2.INTER_NEAREST)
            w, h = template.shape[::-1]
            
            res = cv2.matchTemplate(gray, template,cv2.TM_CCOEFF_NORMED)
            
            ## 35-37 Detect Object
            if between(cap, 35000, 37000):
                min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0]+w,top_left[1]+h)
                cv2.rectangle(frame,top_left,bottom_right,(0,255,255),2)
                """
                threshold = 0.95
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
                """
                cv2.putText(frame, "Template Matching - Detected Correct Orange", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)

            res = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            border_v = 0
            border_h = 0
            border_v = int((frame.shape[1]-res.shape[1])/2)
            border_h = int((frame.shape[0]-res.shape[0])/2)
            res = cv2.copyMakeBorder(res, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, value=[0,0,0])
            res = cv2.resize(res, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            res = (255-res)     
            
            ## Likelihood Map
            if between(cap, 37000, 40000):
                frame = res
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.putText(frame, "Template Matching - Grayscale Likelihood Map", (50,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
        
        out.write(frame)
        cv2.imshow('Frame', frame)

    # Break the loop
    else:
        break

# When everything done, release the video capture and writing object
cap.release()
#out.release()

## 40-50
cap = cv2.VideoCapture('input2.mp4')
#out = cv2.VideoWriter('output2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while(cap.isOpened()):
    ret, frame = cap.read() 
    if ret:
        ## Resize Video
        frame = cv2.resize(frame,(frame_width,frame_height),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        if cv2.waitKey(28) & 0xFF == ord('q'):
            break
        if between(cap, 10000, 60000):
            break
        
        ## 0-5 Track orange
        if between(cap, 000, 5000):
            x_offset =150
            y_offset = 0
            
            roi = frame[y_offset: 360,x_offset: 480]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=15, minRadius=30, maxRadius=40)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x,y,r) in circles:
                    cv2.circle(roi, (x,y), r, (36,255,12), 3)
                    
            cv2.putText(frame, "Object Tracking and Replacement", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)

        ## 5-10 Replace Tracked Orange
        if between(cap, 5000, 10000):
            x_offset = 150
            y_offset = 0
            
            roi = frame[y_offset: 360,x_offset: 480]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=15, minRadius=30, maxRadius=40)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x,y,r) in circles:
                    cv2.circle(roi, (x,y), r, (36,255,12), 3)
                    emoji = cv2.imread("emoji.jpg")
                    emoji = cv2.resize(emoji,(r*2,r*2),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                    
                    if y+y_offset-int(r) < 0:
                        frame[0:y+y_offset+int(r), x+x_offset-int(r):x+x_offset+int(r)] = emoji[int(r)-y-y_offset:2*int(r),:]
                    else:
                        frame[y+y_offset-int(r):y+y_offset+int(r), x+x_offset-int(r):x+x_offset+int(r)] = emoji

            cv2.putText(frame, "Object Tracking and Replacement", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
        
        out.write(frame)
        cv2.imshow('Frame',frame)

    else:
        break

# Release everything if job is finished
cap.release()
#out.release()

## 50-60
cap = cv2.VideoCapture('input1.mp4')
#out = cv2.VideoWriter('output5.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while(cap.isOpened()):
    ret, frame = cap.read() 
    if ret:
        ## Resize Video
        frame = cv2.resize(frame,(frame_width,frame_height),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        if cv2.waitKey(28) & 0xFF == ord('q'):
            break
        if between(cap, 10000, 60000):
            break
        
        ## 50-60
        if between(cap, 000, 10000):
            canny_output = cv2.Canny(frame,100,100)
            contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            contours_poly = [None]*len(contours)
            boundRect = [None]*len(contours)
            centers = [None]*len(contours)
            radius = [None]*len(contours)
            for i, c in enumerate(contours):
                contours_poly[i] = contours[i]
                boundRect[i] = cv2.boundingRect(contours_poly[i])
                centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

            for i in range(len(contours)):
                if (radius[i]>30 and radius[i]<200):
                    color = (255, 0, 0)
                    cv2.drawContours(frame, contours_poly, i, color)
                    cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), \
                      (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
                    cv2.circle(frame, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
            cv2.putText(frame, "Detecting Different Shape Objects", (20,frame_height-50) , font, 0.6, (255,255,255) , 1, cv2.LINE_4)
        
        out.write(frame)
        cv2.imshow('Frame',frame)

    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

