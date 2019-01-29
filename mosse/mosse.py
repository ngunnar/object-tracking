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

class MOSSE():
    def __init__(self, center_x, center_y, width, height, n, l):
        # Class variables
        self.g_2D = None # Desired response
    
        self.A = None # Filter numerator
        self.B = None # Filter denominator
    
        self.search_size = None # 

        self.DX = []
        self.DY = []
        self.PX = []
        self.PY = []

        self.center_coord = [center_x, center_y] # Center coordinate for target [x, y]
        self.target_size = [width, height] # Size of target [width, height]
        self.search_size = [1.5 * width, 1.5 * height]
        self.l = l
        self.n = n

    def get_rectangle(self):
        return (self.center_coord[0] - int(self.target_size[0]/2), self.center_coord[1] - int(self.target_size[1]/2), self.target_size[0], self.target_size[1])

    def cv2_resize(self, x_img):
        return cv2.resize(x_img, (self.target_size[0], self.target_size[1]))

    def get_img_shape(self):
        return (self.target_size[1], self.target_size[0])

    def set_filter(self, X):
        X_conj = np.conjugate(X)
        self.A = np.fft.fft(self.g_2D.reshape(-1,1)) * X_conj
        self.B = X * X_conj + self.l

    def initialize(self, init_img):  
        x_2D = init_img[self.center_coord[1]-int(self.target_size[1]/2):self.center_coord[1]+int(self.target_size[1]/2),
                        self.center_coord[0]-int(self.target_size[0]/2):self.center_coord[0]+int(self.target_size[1]/2)]
        x_2D = rgb2gray(x_2D)
        x_2D_norm = pre_process(self.cv2_resize(x_2D))

        x = x_2D_norm.reshape(-1,1) # reshape to a singel row
        X = np.fft.fft(x)

        # Create Desired Responses, Feature * Filters = Responses
        #Parameters to set
        mu = [self.center_coord[0], self.center_coord[1]]
        covariance = [[self.target_size[0]**2/3, 0], [0, self.target_size[1]**2/3]]

        #Create grid and multivariate normal
        y = np.linspace(self.center_coord[1]-int(self.target_size[1]/2),self.center_coord[1]+int(self.target_size[1]/2),x_2D_norm.shape[0])
        x = np.linspace(self.center_coord[0]-int(self.target_size[0]/2),self.center_coord[0]+int(self.target_size[0]/2),x_2D_norm.shape[1])

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
        return x_2D, x_2D_norm, X

    def test(self, x):
        # Filter
        F = self.A / self.B
        f = np.real(np.fft.ifft(np.conjugate(F)))       
        X = np.fft.fft(x)
        R = F * X

        r = np.real(np.fft.ifft(R))
        r = r.reshape(self.get_img_shape())
        r = linear_mapping(r)

        max_value = np.max(r)
        max_pos = np.where(r == max_value)
        
        _, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(x.reshape(self.get_img_shape()), cmap = plt.get_cmap('gray'))
        ax1.set_title("Template image x")

        ax2.imshow(f.reshape(self.get_img_shape()), cmap = plt.get_cmap('gray'))
        ax2.set_title("Filter f")

        ax3.imshow(r, cmap = plt.get_cmap('gray'))
        ax3.plot(max_pos[1], max_pos[0], 'bo')
        ax3.set_title("Convolution result r")
        plt.show()

    # pre train the filter on the first frame...
    def train_filter(self, x_2D, x_2D_norm, num_training, rotate = False):
        # pre-process img..
        G = np.fft.fft(self.g_2D.reshape(-1,1))
        x = x_2D_norm.reshape(-1,1)
        X = np.fft.fft(x)
        X_conj = np.conjugate(X)
        A = G * X_conj
        B = X * X_conj + self.l
        for _ in range(num_training):
            if rotate:
                x = pre_process(self.cv2_resize(random_warp(x_2D))).reshape(-1,1)
            else:                
                x = pre_process(self.cv2_resize(x_2D)).reshape(-1,1)
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
        x_2D = gray_frame[y_min:self.center_coord[1] + int(self.target_size[1] / 2),
                           x_min:self.center_coord[0] + int(self.target_size[0] / 2)]
        # Preprocess
        x_2D_norm = pre_process(self.cv2_resize(x_2D))

        # Convert to 1-D vector
        x = x_2D_norm.reshape(-1,1)

        # Convolution between test image and filter F(y) = F(p) * F_1
        F = self.A/self.B
        X = np.fft.fft(x)
        Y = X * F        

        # Inverse fourier transform to get result
        y = np.real(np.fft.ifft(Y))
        y_2D = y.reshape(self.get_img_shape())
        y_2D = linear_mapping(y_2D)

        # Get translation
        dx, dy, max_value, max_pos = self._get_translation(y_2D)

        if run_debug:
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
            ax1.imshow(x_2D, cmap = plt.get_cmap('gray'))
            ax1.set_title("Template image x")

            ax2.imshow(x_2D_norm.reshape(self.get_img_shape()), cmap = plt.get_cmap('gray'))
            ax2.set_title("Pre processed template image x")

            f = np.real(np.fft.ifft(np.conjugate(F)))
            ax3.imshow(f.reshape(self.get_img_shape()), cmap = plt.get_cmap('gray'))
            ax3.set_title("Filter f")

            ax4.imshow(y_2D, cmap = plt.get_cmap('gray'))
            # ax4.arrow(x_2D.shape[1]/2, x_2D.shape[0]/2, dy, dx,
            #         head_width=0.05, head_length=1, fc='k', color='red')
            ax4.plot(x_2D.shape[1]/2, x_2D.shape[0]/2, 'ro')
            ax4.plot(x_2D.shape[1]/2 + dy, x_2D.shape[0]/2 + dx, 'bo')
            ax4.set_title("Convolution result y. Max val: {0}".format(max_value))
            plt.show()
            print("Max_value: {0}".format(max_value))
            print("Max_pos_x: {0}".format(max_pos[0]))
            print("Max_pos_y: {0}".format(max_pos[1]))
            print("center_x: {0}".format(self.center_coord[0]))
            print("center_y: {0}".format(self.center_coord[1]))
            print("center_x i: {0}".format(x_2D.shape[0]/2))
            print("center_y i: {0}".format(x_2D.shape[1]/2))
            print("dx_i: {0}".format(dx))
            print("dy_i: {0}".format(dy))
            print()
        
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
        z = pre_process(self.cv2_resize(z_img))
        
        # Convert to 1-D vector
        z = z.reshape(-1,1)       
        
        Z = np.fft.fft(z)
        Y = X * F    

        # Online update
        self._online_update(Z, Y)
        return

    def _get_translation(self, y_2D):
        max_value = np.max(y_2D)
        max_pos = np.where(y_2D == max_value)
        dx = int(np.mean(max_pos[1]) - self.target_size[0]/2)
        dy = int(np.mean(max_pos[0]) - self.target_size[1]/2)
        return dx, dy, max_value, max_pos
    
    def _online_update(self, Z, Y):
        #return
        Z_conj = np.conjugate(Z)
        self.A = (1 - self.n) * self.A + self.n * (Y * Z_conj)
        self.B = (1 - self.n) * self.B + self.n * (Z * Z_conj + self.l)
        return