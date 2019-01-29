import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_utils import rgb2gray, pre_process, linear_mapping, window_func_2d, random_warp

DEBUG = True

class SCCF():
    def __init__(self, center_x, center_y, width, height, n, l):
        # Class variables
        self.g_2D = None # Desired response
    
        self.A = None # Filter numerator
        self.B = None # Filter denominator
    
        self.DX = []
        self.DY = []
        self.PX = []
        self.PY = []

        self.center_coord = [center_x, center_y] # Center coordinate for target [x, y]
        self.target_size = [width, height] # Size of target [width, height]
        self.search_size = [1.5 * width, 1.5 * height] # Size of search area [width, height]
        self.l = l
        self.n = n

    def get_target_rectangle(self):
        return (self.center_coord[0] - int(self.target_size[0]/2), self.center_coord[1] - int(self.target_size[1]/2), self.target_size[0], self.target_size[1])

    def get_search_rectangle(self):
        return (self.center_coord[0] - int(self.search_size[0]/2), self.center_coord[1] - int(self.search_size[1]/2), int(self.search_size[0]), int(self.search_size[1]))

    def cv2_target_resize(self, x_img):
        return cv2.resize(x_img, (self.target_size[0], self.target_size[1]))

    def cv2_search_resize(self, x_img):
        return cv2.resize(x_img, (int(self.search_size[0]), int(self.search_size[1])))

    def get_target_shape(self):
        return (self.target_size[1], self.target_size[0])

    def get_search_shape(self):
        return (int(self.search_size[1]), int(self.search_size[0]))

    def set_filter(self, target_2D_norm):
        target_2D_pad = np.pad(target_2D_norm, [(int((self.search_size[1] - self.target_size[1])/2), ), (int((self.search_size[0] - self.target_size[0])/2), )], mode='constant')
        target_2D_pad = self.cv2_search_resize(target_2D_pad)
        g = self.g_2D.reshape(-1,1)
        X = np.fft.fft(target_2D_pad.reshape(-1,1))
        X_conj = np.conjugate(X)
        self.A = np.fft.fft(g) * X_conj
        self.B = X * X_conj + self.l

    def initialize(self, init_img):  
        target_2D = init_img[self.center_coord[1]-int(self.target_size[1]/2):self.center_coord[1]+int(self.target_size[1]/2),
                        self.center_coord[0]-int(self.target_size[0]/2):self.center_coord[0]+int(self.target_size[1]/2)]

        search_2D = init_img[self.center_coord[1]-int(self.search_size[1]/2):self.center_coord[1]+int(self.search_size[1]/2),
                        self.center_coord[0]-int(self.search_size[0]/2):self.center_coord[0]+int(self.search_size[1]/2)]      
        target_2D = rgb2gray(target_2D)
        search_2D = rgb2gray(search_2D)
        target_2D_norm = pre_process(self.cv2_target_resize(target_2D))
        search_2D_norm = pre_process(self.cv2_search_resize(search_2D))

        # Create Desired Responses, Feature * Filters = Responses
        #Parameters to set
        mu = [self.center_coord[0], self.center_coord[1]]
        covariance = [[self.target_size[0]**2/10, 0], [0, self.target_size[1]**2/10]]

        #Create grid and multivariate normal
        y = np.linspace(self.center_coord[1]-int(self.search_size[1]/2),self.center_coord[1]+int(self.search_size[1]/2),search_2D_norm.shape[0])
        x = np.linspace(self.center_coord[0]-int(self.search_size[0]/2),self.center_coord[0]+int(self.search_size[0]/2),search_2D_norm.shape[1])

        X_mesh, Y_mesh = np.meshgrid(x,y)
        pos = np.empty(X_mesh.shape + (2,))
        pos[:, :, 0] = X_mesh 
        pos[:, :, 1] = Y_mesh

        r = multivariate_normal(mu, covariance)
        self.g_2D = linear_mapping(r.pdf(pos))
        if DEBUG:
            max_value = np.max(self.g_2D)
            max_pos = np.where(self.g_2D == max_value)           

            plt.imshow(self.g_2D, cmap = plt.get_cmap('gray'))
            plt.title('G 2D')
            print(np.median(max_pos[1]))
            print(np.median(max_pos[0]))
            print(max_pos[0])
            print(max_pos)
            plt.plot(np.median(max_pos[1]), np.median(max_pos[0]), 'bo')
            plt.show()

            #Make a 3D plot
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X_mesh, Y_mesh, self.g_2D,cmap='viridis',linewidth=0)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis'),
            ax.set_zlabel('Z axis')
            plt.show()
        return target_2D, target_2D_norm

    def test(self, frame):
        #target_2D_pad = np.pad(target_2D_norm, [(int((self.search_size[1] - self.target_size[1])/2), ), (int((self.search_size[0] - self.target_size[0])/2), )], mode='constant')
        #target_2D_pad = self.cv2_search_resize(target_2D_pad)
        search_2D = frame[self.center_coord[1]-int(self.search_size[1]/2):self.center_coord[1]+int(self.search_size[1]/2),
                         self.center_coord[0]-int(self.search_size[0]/2):self.center_coord[0]+int(self.search_size[1]/2)]       
        search_2D = rgb2gray(search_2D)
        #search_2D_norm = pre_process(self.cv2_search_resize(search_2D))
        search_2D_norm = pre_process(self.cv2_search_resize(search_2D))
        # Filter        
        F = self.A / self.B
        f = np.real(np.fft.ifft(np.conjugate(F)))       
        X = np.fft.fft(search_2D_norm.reshape(-1,1))
        R = F * X
        x = np.real(np.fft.ifft(X))

        r = np.real(np.fft.ifft(R)).reshape(self.get_search_shape())
        r = linear_mapping(r)

        max_value = np.max(r)
        max_pos = np.where(r == max_value)
        
        _, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(x.reshape(self.get_search_shape()), cmap = plt.get_cmap('gray'))
        ax1.set_title("Template image x")

        ax2.imshow(f.reshape(self.get_search_shape()), cmap = plt.get_cmap('gray'))
        ax2.set_title("Filter f")

        ax3.imshow(r, cmap = plt.get_cmap('gray'))
        ax3.plot(max_pos[1], max_pos[0], 'bo')
        ax3.set_title("Convolution result r")
        plt.show()

    # pre train the filter on the first frame...
    def train_filter(self, target_2D, target_2D_norm, num_training, rotate = False):
        # pre-process img..
        # pre-process
        # pad
        target_2D_pad = np.pad(target_2D_norm, [(int((self.search_size[1] - self.target_size[1])/2), ), (int((self.search_size[0] - self.target_size[0])/2), )], mode='constant')
        target_2D_pad = self.cv2_search_resize(target_2D_pad)        
        
        G = np.fft.fft(self.g_2D.reshape(-1,1))
        x = target_2D_pad.reshape(-1,1)
        X = np.fft.fft(x)
        X_conj = np.conjugate(X)
        A = G * X_conj
        B = X * X_conj + self.l
        for _ in range(num_training):
            if rotate:
                t_2D_norm = pre_process(self.cv2_target_resize(random_warp(target_2D)))
            else: 
                t_2D_norm = pre_process(self.cv2_target_resize(target_2D))            
            
            t_2D_pad = np.pad(t_2D_norm, [(int((self.search_size[1] - self.target_size[1])/2), ), (int((self.search_size[0] - self.target_size[0])/2), )], mode='constant')
            x = self.cv2_search_resize(t_2D_pad).reshape(-1,1)
            X = np.fft.fft(x)
            X_conj = np.conjugate(X)
            A += X_conj * G
            B += X_conj * X  + self.l

        self.A = A
        self.B = B

    def run(self, frame, run_debug = False):
        # Convert image to grayscale
        gray_frame = rgb2gray(frame)

        # Select template
        y_min = np.max([self.center_coord[1] - int(self.target_size[1] / 2),0])
        x_min = np.max([self.center_coord[0] - int(self.target_size[0] / 2),0])
        target_2D = gray_frame[y_min:self.center_coord[1] + int(self.target_size[1] / 2),
                                 x_min:self.center_coord[0] + int(self.target_size[0] / 2)]
        # Preprocess
        target_2D_norm = pre_process(self.cv2_target_resize(target_2D))
        
        target_2D_pad = np.pad(target_2D_norm, [(int((self.search_size[1] - self.target_size[1])/2), ), (int((self.search_size[0] - self.target_size[0])/2), )], mode='constant')
        target_2D_pad = self.cv2_search_resize(target_2D_pad)

        # Convert to 1-D vector
        x = target_2D_pad.reshape(-1,1)

        # Convolution between test image and filter F(y) = F(p) * F_1
        F = self.A/self.B
        X = np.fft.fft(x)
        Y = X * F        

        # Inverse fourier transform to get result
        y = np.real(np.fft.ifft(Y))
        y_2D = y.reshape(self.get_search_shape())
        y_2D = linear_mapping(y_2D)

        # Get translation
        dx, dy, max_value, max_pos = self._get_translation(y_2D)

        if run_debug:
            print("Max_pos_x: {0}".format(max_pos[1]))
            print("Max_pos_y: {0}".format(max_pos[0]))
            print("center_x: {0}".format(self.center_coord[0]))
            print("center_y: {0}".format(self.center_coord[1]))
            print("center_x_i: {0}".format(self.search_size[0]/2))
            print("center_y_i: {0}".format(self.search_size[1]/2))
            print("dx_i: {0}".format(dx))
            print("dy_i: {0}".format(dy))
            print()
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
            ax1.imshow(target_2D, cmap = plt.get_cmap('gray'))
            ax1.set_title("Template image x")

            ax2.imshow(target_2D_pad.reshape(self.get_search_shape()), cmap = plt.get_cmap('gray'))
            ax2.set_title("Pre processed template image x")

            f = np.real(np.fft.ifft(np.conjugate(F)))
            ax3.imshow(f.reshape(self.get_search_shape()), cmap = plt.get_cmap('gray'))
            ax3.set_title("Filter f")

            ax4.imshow(y_2D, cmap = plt.get_cmap('gray'))
            # ax4.arrow(target_2D_pad.shape[1]/2, target_2D_pad.shape[0]/2, dy, dx,
            #         head_width=0.05, head_length=1, fc='k', color='red')
            ax4.plot(target_2D_pad.shape[1]/2, target_2D_pad.shape[0]/2, 'ro')
            ax4.plot(max_pos[1], max_pos[0], 'bo')
            ax4.set_title("Convolution result y. Max val: {0}".format(max_value))        
            plt.show()

        
        # Update center points of template with momentum
        beta = 0
        self.center_coord[0] = int(self.center_coord[0] + (1 - beta) * dx)
        self.center_coord[1] = int(self.center_coord[1] + (1 - beta) * dy)    
        
        # Save
        self.DX.append(dx)
        self.DY.append(dy)
        self.PX.append(self.center_coord[0])
        self.PY.append(self.center_coord[1])
        
        # get updated template z
        y_min = np.max([self.center_coord[1] - int(self.target_size[1] / 2),0])
        x_min = np.max([self.center_coord[0] - int(self.target_size[0] / 2),0])
        
        z_img = gray_frame[y_min:self.center_coord[1] + int(self.target_size[1] / 2),
                           x_min:self.center_coord[0] + int(self.target_size[0] / 2)]                            
        # Preprocess
        z_norm = pre_process(self.cv2_target_resize(z_img))
        
        z_2D_pad = np.pad(z_norm, [(int((self.search_size[1] - self.target_size[1])/2), ), (int((self.search_size[0] - self.target_size[0])/2), )], mode='constant')
        z_2D_pad = self.cv2_search_resize(z_2D_pad)


        # Convert to 1-D vector
        z = z_2D_pad.reshape(-1,1)       
        
        Z = np.fft.fft(z)
        Y = X * F    

        # Online update
        self._online_update(Z, Y)
        return

    def _get_translation(self, y_2D):
        #max_y, max_x = np.unravel_index(np.argmax(y_2D), y_2D.shape)  # find the match
        max_value = np.max(y_2D)
        max_pos = np.where(y_2D == max_value)
        dx = int(np.mean(max_pos[1]) - self.search_size[0]/2)
        dy = int(np.mean(max_pos[0]) - self.search_size[1]/2)
        return dx, dy, max_value, max_pos
    
    def _online_update(self, Z, Y):
        #return
        Z_conj = np.conjugate(Z)
        self.A = (1 - self.n) * self.A + self.n * (Y * Z_conj)
        self.B = (1 - self.n) * self.B + self.n * (Z * Z_conj + self.l)
        return