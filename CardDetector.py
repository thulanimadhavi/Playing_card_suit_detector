############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a PiCamera video feed.
#

# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards
import VideoStream


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings


## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed


## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:


# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
path2 = (path + '\Card_Imgs')
train_ranks = Cards.load_ranks()
train_suits = Cards.load_suits()


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.
img = cv2.imread('queen.jpg', cv2.IMREAD_UNCHANGED)
dimensions = img.shape
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)



# Begin capturing frames


    # Grab frame from video stream
image = cv2.imread("ace2.jpg")
resized_image = cv2.resize(image,(599,900))
dimensions = resized_image.shape
height = resized_image.shape[0]
width = resized_image.shape[1]
channels = resized_image.shape[2]
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)

    # Start timer (for calculating frame rate)


    # Pre-process camera image (gray, blur, and threshold it)
pre_proc = Cards.preprocess_image(image)

	
    # Find and sort the contours of all cards in the image (query cards)
cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)


    # If there are no contours, do nothing
if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
    cards = []
    k = 0


        # For each contour detected:
    for i in range(len(cnts_sort)):
        if (cnt_is_card[i] == 1):

                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.
            cards.append(Cards.preprocess_card(cnts_sort[i],image))
            print(cards)

                # Find the best rank and suit match for the card.
            cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

                # Draw center point and match result on the image.
            image = Cards.draw_results(image, cards[k])
            k = k + 1

	    
        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
    if (len(cards) != 0):
        temp_cnts = []
        for i in range(len(cards)):
            temp_cnts.append(cards[i].contour)
        cv2.drawContours(image,temp_cnts, -1, (0,0,255), 2)

        
        
    # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
    # so the first time this runs, framerate will be shown as 0.


    # Finally, display the image with the identified cards!
cv2.imshow("Card Detector",image)

    # Calculate framerate

    
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
cv2.waitKey(0)

        

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()


