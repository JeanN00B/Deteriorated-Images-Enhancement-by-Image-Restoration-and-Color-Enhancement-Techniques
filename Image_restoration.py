import cv2 as cv
import numpy as np
import math
import time
import scipy.io
import os
import glob
from numpy.fft import fft2, ifft2
from skimage.util import img_as_float
from skimage import data, io, color
from scipy.signal import gaussian
from matplotlib import pyplot as plt



#============================================================================#
#========================== RESOLUTION IMPROVEMENT ==========================#
#============================================================================#
"""
 1. Read the picture
 2. Apply algorithm to a copy 
 3. Save that copy as the same name + ReImprov
"""
def rescale_image(image, scale = 2):
       # first create a copy that is image*scale matrix
       image = image.astype(np.float)
       row = image.shape[0]*scale
       col = image.shape[1]*scale

       scaled_image = np.zeros((row, col, 3)) #RGB MATRIX

       for i in range(row):
              for j in range(col):
                     x_val = i / scale
                     y_val = j / scale

                     x1, y1 = int(x_val), int(y_val) # 4 esquinas más cercanas
                     x2, y2 = (x1), (y1+1)
                     x3, y3 = (x1+1),  (y1)
                     x4, y4 = (x1+1), (y1+1)
                     
                     u = x_val - int(x_val) # diferencial
                     v = y_val - int(y_val)

                     if x4 >= image.shape[0]: # problemas llegando a los límites
                            x4 = image.shape[0] - 1
                            x2, x1, x3 = x4, (x4-1), (x4-1)

                     if y4 >= image.shape[1]: # problemas llegando a los límites
                            y4 = image.shape[1] - 1
                            y2, y1, y3 = (y4-1), (y4-1), y4

                     # For R G B channels
                     scaled_image[i, j][0] = (1-u)*(1-v)*int(image[x1, y1][0]) + (1-u)*v*int(image[x2,y2][0]) + u*(1-v)*int(image[x3,y3][0]) + u*v*int(image[x4,y4][0])
                     scaled_image[i, j][1] = (1-u)*(1-v)*int(image[x1, y1][1]) + (1-u)*v*int(image[x2,y2][1]) + u*(1-v)*int(image[x3,y3][1]) + u*v*int(image[x4,y4][1])
                     scaled_image[i, j][2] = (1-u)*(1-v)*int(image[x1, y1][2]) + (1-u)*v*int(image[x2,y2][2]) + u*(1-v)*int(image[x3,y3][2]) + u*v*int(image[x4,y4][2])                     

                     scaled_result = scaled_image.astype(np.uint8)
       return scaled_result

#============================================================================#
#============================= IMAGE RESTORATION ============================#
#============================================================================#
"""
1. Load image
2. Apply adaptative local noise reduction filter
3. Test Wiener filter
4. Save and enjoy an improved picture ヾ(•ω•`)o
"""

#336 (338 DE 1022) book pdf page

#Histogram
def variance_matrix(image): # 'image' means matrix
       colors = [0]*256 #from 0 to 255
       for i in range(len(image)): #separate pixels values into an 256 lenght array
              for j in range(len(image[0])):
                     colors[int(image[i][j])] +=1 # number of pixels to each pixel value
       normalized_hist_array = np.array(colors)/(len(image)*len(image[0])) # normalization

       m_value = 0
       for i in range(0, 255):
              m_value += i*normalized_hist_array[i]

       variance = 0
       for i in range(0, 255):
              variance += (i - m_value)*(i - m_value)*normalized_hist_array[i]

       return variance

#Get variance from image Sxy + intensity to save computational sources :)
def variance_intensity_Sxy(picture, pos_x, pos_y, Sxy_dimension = 3):
       Sxy = np.zeros((Sxy_dimension,Sxy_dimension))
       Sxy_intensity = 0
       num_pixels = 0

       for x in range(int(-((Sxy_dimension-1)/2)), int((Sxy_dimension-1)/2)): # 5*5 Sxy space
              for y in range(int(-((Sxy_dimension-1)/2)), int((Sxy_dimension-1)/2)):
                     try:
                            Sxy[x,y] = picture[x+pos_x][y+pos_y] #Save Sxy temporally
                            if (x == 0) and (y == 0): # On the pixel we want to change, we exclude it
                                   continue
                            else:
                                   Sxy_intensity += picture[x+pos_x][y+pos_y]
                     except:
                            Sxy[x,y] = 0 #Not the most appropiate while calculating img boundaries pixels val

       Sxy_intensity = float(Sxy_intensity/(Sxy_dimension*Sxy_dimension - 1))
       Sxy_variance = variance_matrix(Sxy)

       return Sxy_variance, Sxy_intensity


def adaptative_loc_noise_reduc(image_RGB, Sxy_dimension = 3):
       filtered_image = np.zeros_like(image_RGB)
       filtered_image_R = np.zeros_like(image_RGB[:,:,0])
       filtered_image_G = np.zeros_like(image_RGB[:,:,1])
       filtered_image_B = np.zeros_like(image_RGB[:,:,2])

       img_var_R = variance_matrix(image_RGB[:,:,0])
       img_var_G = variance_matrix(image_RGB[:,:,1])
       img_var_B = variance_matrix(image_RGB[:,:,2])

       for x in range(len(image_RGB)):
              for y in range(len(image_RGB[0])):
                     Sxy_var, intensity_R = variance_intensity_Sxy(image_RGB[:,:,0], x, y, Sxy_dimension)
                     Sxy_var, intensity_G = variance_intensity_Sxy(image_RGB[:,:,1], x, y, Sxy_dimension)
                     Sxy_var, intensity_B = variance_intensity_Sxy(image_RGB[:,:,2], x, y, Sxy_dimension)

                     if img_var_R > Sxy_var:
                            division_R = 1
                     else:
                            division_R = float(img_var_R/Sxy_var)

                     if img_var_G > Sxy_var:
                            division_G = 1
                     else:
                            division_G = float(img_var_G/Sxy_var)

                     if img_var_B > Sxy_var:
                            division_B = 1
                     else:
                            division_B = float(img_var_B/Sxy_var)

                     filtered_image_R[x,y] = (image_RGB[x,y][0] - division_R*float(image_RGB[x,y][0] - intensity_R)).astype(np.uint8)
                     filtered_image_G[x,y] = (image_RGB[x,y][1] - division_G*float(image_RGB[x,y][1] - intensity_G)).astype(np.uint8)
                     filtered_image_B[x,y] = (image_RGB[x,y][2] - division_B*float(image_RGB[x,y][2] - intensity_B)).astype(np.uint8)

       filtered_image[:,:,0] = filtered_image_R
       filtered_image[:,:,1] = filtered_image_G
       filtered_image[:,:,2] = filtered_image_B

       return filtered_image


# Wiener filter implementation from github ;3

def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return dummy

# Creates the Gaussian Kernel that goes into the wiener filter
def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

#============================================================================#
#============================= COLOR ENHACEMENT =============================#
#============================================================================#
"""
1. From RGB to CMYK and inverse
2. From RGB to HSI and inverse
3. Library from BGR to RGB :3
"""

def CMYK_RGB(image, CMYK = False):
       if CMYK == True: #convert to CMYK
              CMYK_matrix = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
              for x in range(len(image)):
                     for y in range(len(image[0])):
                            #From RGB to CMYK
                            C_chan = 1- image[x,y][0]/255 
                            M_chan = 1- image[x,y][1]/255
                            Y_chan = 1- image[x,y][2]/255

                            k_val = min(C_chan, M_chan, Y_chan)
                            CMYK_matrix[x,y][0] = 255*((C_chan - k_val)/(1 - k_val))
                            CMYK_matrix[x,y][1] = 255*((M_chan - k_val)/(1 - k_val))
                            CMYK_matrix[x,y][2] = 255*((Y_chan - k_val)/(1 - k_val))
                            CMYK_matrix[x,y][3] = 255*k_val
              return CMYK_matrix

       else: #Convert to RGB
              RGB_matrix = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
              for x in range(len(image)):
                     for y in range(len(image[0])):
                            #From RGB to CMYK
                            C_chan = image[x,y][0]/255
                            M_chan = image[x,y][1]/255
                            Y_chan = image[x,y][2]/255
                            k_val = image[x,y][3]/255

                            RGB_matrix[x,y][0] = 255 * (1 - C_chan) * (1 - k_val)
                            RGB_matrix[x,y][1] = 255 * (1 - M_chan) * (1 - k_val)
                            RGB_matrix[x,y][2] = 255 * (1 - Y_chan) * (1 - k_val)
              return RGB_matrix


def HSI_RGB(image, HSI = False):
       if HSI == True:
              HSI_picture = np.zeros_like(image, dtype = np.uint8)
              for x in range(len(image)):
                     for y in range(len(image[0])):
                            R_col = image[x, y][0]/255
                            G_col = image[x, y][1]/255
                            B_col = image[x, y][2]/255
                            min_val = min(R_col, G_col, B_col)

                            ################################################
                            saturation = (1 - (3/(R_col + G_col + B_col))*min_val)
                            intensity = ((R_col + G_col + B_col)/3)

                            if R_col == G_col == B_col:
                                   hue_angle = 90
                            else:
                                   hue_angle = math.acos( (((R_col - G_col) + (R_col - B_col))/2)/(0.000000000001 + math.sqrt((R_col - G_col)**2 + (R_col - B_col)*(G_col - B_col))))*(360/(2*math.pi))
                         
                            if B_col <= G_col:
                                   hue = round((hue_angle/360), 6) #Set to a 0-1 interval
                            else:
                                   hue = round((360 - hue_angle)/360, 6)
                             
                            #Store every value in range of 0 - 255

                            HSI_picture[x,y] = [hue*255, saturation*255, intensity*255]
              return HSI_picture

       elif HSI == False:
              RGB_picture = np.zeros_like(image, dtype = np.uint8)
              for x in range(len(image)):
                     for y in range(len(image[0])):
                            H_col = image[x, y][0]/255 #Values between 0 - 1
                            S_col = image[x, y][1]/255
                            I_col = image[x, y][2]/255

                            H_col =  H_col*360 #Transformed to 360 degrees range
                            if (H_col >= 0) and (H_col < 120):
                                   R_col = I_col*((1 + (S_col*math.cos(H_col))/(0.000000000001 + math.cos(math.radians(60 - H_col)))))
                                   B_col = I_col*(1 - S_col)
                                   G_col = 3*I_col-(R_col + B_col)
                                   RGB_picture[x,y] = [R_col*255, G_col*255, B_col*255]

                            elif (H_col >= 120) and (H_col < 240):
                                   H_col = H_col - 120
                                   G_col = I_col*((1 + (S_col*math.cos(H_col))/(0.000000000001 + math.cos(math.radians(60 - H_col)))))
                                   R_col = I_col*(1 - S_col)
                                   B_col = 3*I_col-(R_col + G_col)
                                   RGB_picture[x,y] = [R_col*255, G_col*255, B_col*255]

                            elif (H_col >= 240) and (H_col <= 360):
                                   H_col = H_col - 240
                                   B_col = I_col*((1 + (S_col*math.cos(H_col))/(0.000000000001 + math.cos(math.radians(60 - H_col)))))
                                   G_col = I_col*(1 - S_col)
                                   R_col = 3*I_col-(B_col + G_col)
                                   RGB_picture[x,y] = [R_col*255, G_col*255, B_col*255]
              return RGB_picture



#==========================================================================================================#
#============================================= IMPLEMENTATION =============================================#
#==========================================================================================================#
def main_menu():
       print("»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«\n»Image enhacement of every picture in the directory this script is executed«\n»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«»«\n")
       action = input("(1) Color correction only (fastest option)\n(2) Standar image enhacement (resolution and color)\n(3) Noise reduction (filter)\nChoose an option: ")
       if action not in ["1", "2", "3"]:
              print("Unknow action, returning to main menu")
              time.sleep(1)
              os.system("cls")
              main_menu()
       else:
              if action == "1":

                     path = os.getcwd()
                     for img in glob.glob(path + '\*.jpg'):
                            print("Reading Image...")
                            image = cv.imread(os.path.basename(img))
                            #Color equalization only
                            R_col, G_col, B_col = cv.split(image)
                            R_col_e = cv.equalizeHist(R_col)
                            G_col_e = cv.equalizeHist(G_col)
                            B_col_e = cv.equalizeHist(B_col)
                            corrected_image = cv.merge((R_col_e, G_col_e, B_col_e))
                            print("Image Corrented!")
                            filename = os.path.basename(img)
                            filename = "corrected_" + filename 
                            cv.imwrite(filename, corrected_image)
                            print("Image Saved!")
                     print("Done!")
                     time.sleep(1)
                     os.system("cls")

              elif action == "2":
                     path = os.getcwd()
                     for img in glob.glob(path + '\*.jpg'):
                            print("Reading Image...")
                            image = cv.imread(os.path.basename(img))
                            #First resolution, then color
                            image = rescale_image(image, scale=2)
                            R_col, G_col, B_col = cv.split(image)
                            R_col_e = cv.equalizeHist(R_col)
                            G_col_e = cv.equalizeHist(G_col)
                            B_col_e = cv.equalizeHist(B_col)
                            corrected_image = cv.merge((R_col_e, G_col_e, B_col_e))
                            print("Image Corrented!")
                            filename = os.path.basename(img)
                            filename = "corrected_" + filename 
                            cv.imwrite(filename, corrected_image)
                            print("Image Saved!")
                     print("Done!")
                     time.sleep(1)
                     os.system("cls")

              elif action == "3":
                     path = os.getcwd()
                     for img in glob.glob(path + '\*.jpg'):
                            print("Reading Image...")
                            image = cv.imread(os.path.basename(img))
                            #First ALNF
                            image = adaptative_loc_noise_reduc(image_RGB, Sxy_dimension=3)
                            print("Image Corrented!")
                            filename = os.path.basename(img)
                            filename = "corrected_" + filename 
                            cv.imwrite(filename, corrected_image)
                            print("Image Saved!")
                     print("Done!")
                     time.sleep(1)
                     os.system("cls")

#The magic starts here :)
main_menu()


#===========================================================================================
#Code used to generate report images :)
"""
#Load image to bee processed
pictures_list = ["Fotos antiguas 015.jpg", "Fotos antiguas 006.jpg", "Fotos antiguas 001.jpg", "Cool_face.jpg", "x3_scaled_cool_face.jpg", "Fotos antiguas 022.jpg"]
noisy_image = cv.imread(pictures_list[0])
noisy_image = cv.cvtColor(noisy_image, cv.COLOR_BGR2RGB)
"""
#==========================================================================================================#
#=================================== ALNR FILTER AND COLOR CORRECTION =====================================#
#==========================================================================================================#
"""
ALNR_RGB_image = adaptative_loc_noise_reduc(noisy_image, Sxy_dimension=7)
R_col, G_col, B_col = cv.split(ALNR_RGB_image)
R_col_e = cv.equalizeHist(R_col)
G_col_e = cv.equalizeHist(G_col)
B_col_e = cv.equalizeHist(B_col)
ALNR_RGB_image = cv.merge((R_col_e, G_col_e, B_col_e))

RGB_ALNR_image = noisy_image.copy()
R_col, G_col, B_col = cv.split(RGB_ALNR_image)
R_col_e = cv.equalizeHist(R_col)
G_col_e = cv.equalizeHist(G_col)
B_col_e = cv.equalizeHist(B_col)
RGB_ALNR_image = cv.merge((R_col_e, G_col_e, B_col_e))


plt.subplot(121)
plt.imshow(RGB_ALNR_image)
plt.title("Color corrected image")

plt.subplot(122)
plt.imshow(ALNR_RGB_image)
plt.title("First noise reduction, then color correction")

plt.show()
"""
#==========================================================================================================#
#================================ DIFFERENT KERNERL SIZE ON ALNR FILTER ===================================#
#==========================================================================================================#
"""
scaled_image1 = adaptative_loc_noise_reduc(noisy_image, Sxy_dimension = 3)
scaled_image2 = adaptative_loc_noise_reduc(noisy_image, Sxy_dimension = 5)
scaled_image3 = adaptative_loc_noise_reduc(noisy_image, Sxy_dimension = 7)
plt.subplot(221)
plt.imshow(noisy_image)
plt.title('original picture')

plt.subplot(222)
plt.imshow(scaled_image1)
plt.title('scaled_picture 3 x 3')

plt.subplot(223)
plt.imshow(scaled_image2)
plt.title('scaled_picture 5 x 5')

plt.subplot(224)
plt.imshow(scaled_image3)
plt.title('scaled_picture 7 x 7')
plt.show()
"""

#==========================================================================================================#
#=================================== COLOR CORRECTION CMYK VS RGB VS HSI ==================================#
#==========================================================================================================#
"""
# CMYK
CMYK_correction_img = CMYK_RGB(noisy_image, CMYK=True)
C_col, M_col, Y_col, K_col = cv.split(CMYK_correction_img)
C_col_e = cv.equalizeHist(C_col)
M_col_e = cv.equalizeHist(M_col)
Y_col_e = cv.equalizeHist(Y_col)
K_col_e = cv.equalizeHist(K_col)
CMYK_correction_img1 = cv.merge((C_col_e, M_col, Y_col, K_col))
CMYK_correction_img2 = cv.merge((C_col, M_col_e, Y_col, K_col))
CMYK_correction_img3 = cv.merge((C_col, M_col, Y_col_e, K_col))
CMYK_correction_img4 = cv.merge((C_col, M_col, Y_col, K_col_e))
CMYK_correction_img5 = cv.merge((C_col_e, M_col_e, Y_col_e, K_col_e))
CMYK_correction_img1 = CMYK_RGB(CMYK_correction_img1, CMYK=False)
CMYK_correction_img2 = CMYK_RGB(CMYK_correction_img2, CMYK=False)
CMYK_correction_img3 = CMYK_RGB(CMYK_correction_img3, CMYK=False)
CMYK_correction_img4 = CMYK_RGB(CMYK_correction_img4, CMYK=False)
CMYK_correction_img5 = CMYK_RGB(CMYK_correction_img5, CMYK=False)

plt.subplot(231)
plt.imshow(CMYK_correction_img1)
plt.title('Cyan channel corrected')

plt.subplot(232)
plt.imshow(CMYK_correction_img2)
plt.title('Magenta channel corrected')

plt.subplot(233)
plt.imshow(CMYK_correction_img3)
plt.title('Yellow channel corrected')

plt.subplot(234)
plt.imshow(CMYK_correction_img4)
plt.title('Black channel corrected')

plt.subplot(235)
plt.imshow(CMYK_correction_img5)
plt.title('All channel corrected')
plt.show()

#==========================================================================================================#

# RGB
RGB_correction_img = noisy_image.copy()
R_col, G_col, B_col = cv.split(RGB_correction_img)
R_col_e = cv.equalizeHist(R_col)
G_col_e = cv.equalizeHist(G_col)
B_col_e = cv.equalizeHist(B_col)
RGB_correction_img1 = cv.merge((R_col_e, G_col, B_col))
RGB_correction_img2 = cv.merge((R_col, G_col_e, B_col))
RGB_correction_img3 = cv.merge((R_col, G_col, B_col_e))
RGB_correction_img4 = cv.merge((R_col_e, G_col_e, B_col_e))
plt.subplot(221)
plt.imshow(RGB_correction_img1)
plt.title('Red channel corrected')

plt.subplot(222)
plt.imshow(RGB_correction_img2)
plt.title('Green channel corrected')

plt.subplot(223)
plt.imshow(RGB_correction_img3)
plt.title('Blue channel corrected')

plt.subplot(224)
plt.imshow(RGB_correction_img4)
plt.title('All channels corrected')

plt.show()

#==========================================================================================================#

#HSI
HSI_correction_img = HSI_RGB(noisy_image, HSI=True)
H_col, S_col, I_col = cv.split(HSI_correction_img)
H_col_e = cv.equalizeHist(H_col) 
S_col_e = cv.equalizeHist(S_col)
I_col_e = cv.equalizeHist(I_col)
HSI_correction_img1 = cv.merge((H_col_e, S_col, I_col))
HSI_correction_img2 = cv.merge((H_col, S_col_e, I_col))HSI_correction_img3 = cv.merge((H_col, S_col, I_col_e))
HSI_correction_img4 = cv.merge((H_col_e, S_col_e, I_col_e))
HSI_correction_img1 = HSI_RGB(HSI_correction_img1, HSI=False)
HSI_correction_img2 = HSI_RGB(HSI_correction_img2, HSI=False)
HSI_correction_img3 = HSI_RGB(HSI_correction_img3, HSI=False)
HSI_correction_img4 = HSI_RGB(HSI_correction_img4, HSI=False)
plt.subplot(221)
plt.imshow(HSI_correction_img1)# THE PROBLEM IS IN THE HUE
plt.title('Hue channel corrected')

plt.subplot(222)
plt.imshow(HSI_correction_img2)
plt.title('Saturation channel corrected')

plt.subplot(223)
plt.imshow(HSI_correction_img3)
plt.title('Intensity channel corrected')

plt.subplot(224)
plt.imshow(HSI_correction_img4)
plt.title('All channels corrected')

plt.show()
"""
#==========================================================================================================#
#================================SOME EXTRA ALGORITHM COMPARISONS==========================================#
#==========================================================================================================#

"""
kernel = gaussian_kernel(3)
#wiener_image = wiener_filter(noisy_image, kernel, K = 250)

plt.subplot(131)
plt.imshow(noisy_image)
plt.title('original picture')

plt.subplot(132)
plt.imshow(color_correction_img)
plt.title('Color correction')

plt.subplot(133)
plt.imshow(ALNR_image)
plt.title('Adaptativ local noise reduction filter')

plt.show()
"""
