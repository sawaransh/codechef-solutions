from PIL import Image
import numpy as np
# Do not change the above code. Write your code below

def process_grayscale(img_array):
    # Save grayscale image array
    np.savetxt("grey_output.txt", img_array, fmt="%d")

def process_rgb(img_array):
    # Save RGB image array (flatten last dimension for saving)
    reshaped = img_array.reshape(-1, img_array.shape[-1])
    np.savetxt("rgb_output.txt", reshaped, fmt="%d")

def img_to_array(path):
    # Read image
    img = Image.open(path)
    arr = np.array(img)
    
    # Check dimensions
    if arr.ndim == 2:
        process_grayscale(arr)
    elif arr.ndim == 3:
        process_rgb(arr)
    else:
        raise ValueError("Unsupported image dimensions")

# Driver code
path = input().strip()
img_to_array(path)