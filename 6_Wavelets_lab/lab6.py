import numpy as np
import pywt
import math

def norm_std_map(img):
    output = np.copy(img)
    
    # YOUR CODE HERE
    
    return output


def non_uniform_map(img):
    output = np.copy(img)
    if -np.amin(img) != 0:
        neg_func = 127.5*(img - np.amin(img))/(-np.amin(img))
    else:
        neg_func = 127.5*(img - np.amin(img))
    if np.amax(img) != 0:
        pos_func = 127.5 + 127.5*(img)/(np.amax(img))
    else:
        pos_func = 127.5*(img - np.amin(img))
    output[img < 0] = neg_func[img < 0]
    output[img >= 0] = pos_func[img >= 0]
    
    return output

def map_color(img, n = 0, color_map = np.array):    
    # This first block is to determine wether we have a raw image or the transform
    div = 2**(n)
    ny, nx = np.array(img.shape) / div
    # ny and nx represent the size of the low frequency coefficients 
    ny, nx = int(ny), int(nx)
    
    # Generate output array to work on
    output = np.copy(img)
    
    # First we apply transform to top left coefficient. 
    # Note that if n = 0, we will be getting applying to the whole image
    output[0:ny, 0:nx] = color_map(output[0:ny, 0:nx])
    # Now we will iterate through the number of WT iterations and process the other three regions
    # Note that if n = 0, for loop will not start
    for i in range(n):
        # Apply transform to high-frequency components
        output[0:ny, nx:2*nx] = color_map(output[0:ny, nx:2*nx])
        output[ny:2*ny, nx:2*nx] = color_map(output[ny:2*ny, nx:2*nx])
        output[ny:2*ny, 0:nx] = color_map(output[ny:2*ny, 0:nx])
        
        # Update dimensions
        nx = nx * 2
        ny = ny * 2
        
    output = 255*(output - np.amin(output))/(np.amax(output-np.amin(output)))
    return output

def pywt_analysis(img, n, wavelet = 'haar'):
    ny, nx = img.shape
    output = np.copy(img)
    for i in range(n):
        sub = output[0:ny, 0:nx]
        cA, (cV, cH, cD) = pywt.dwt2(sub, wavelet = wavelet, mode = 'periodization')
        nx = int(nx/2)
        ny = int(ny/2)     
        output[0:ny, 0:nx] = cA
        output[ny:2*ny, 0:nx] = cV
        output[0:ny, nx:2*nx] = cH
        output[ny:2*ny, nx:2*nx] = cD
    return output

def pywt_synthesis(img, n, wavelet = 'pywt'):
    # Get information about the transform (size of the last Wavelet Transform)
    div = 2**(n-1)
    ny, nx = np.array(img.shape) / div
    ny, nx = int(ny), int(nx)
    # Generate output array to work on
    output = np.copy(img)
        
    # Iterate through n
    for i in range(n):
        # Extract coefficients
        cA = output[0:int(ny/2), 0:int(nx/2)]
        cH = output[0:int(ny/2), int(nx/2):nx]
        cV = output[int(ny/2):ny, 0:int(nx/2)]
        cD = output[int(ny/2):ny, int(nx/2):nx]
        # Apply inverse transform
        sub = pywt.idwt2((cA, (cV, cH, cD)), mode = 'periodization', wavelet = wavelet)
        # Replace inverse transform in image
        output[0:ny, 0:nx] = sub
        # Update dimensions
        nx = nx * 2
        ny = ny * 2
    
    return output

def snr_db(img, orig):
    nmse = 10*np.log10(np.sum(orig**2)/np.sum((img-orig)**2))
    return nmse
