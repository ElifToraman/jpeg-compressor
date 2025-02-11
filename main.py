import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import pickle

# Step 1: (RGB -> YCbCr)

def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    Y  =  0.299    * img[:,:,0] + 0.587    * img[:,:,1] + 0.114    * img[:,:,2]
    Cb = -0.168736 * img[:,:,0] - 0.331264 * img[:,:,1] + 0.5      * img[:,:,2] + 128
    Cr =  0.5      * img[:,:,0] - 0.418688 * img[:,:,1] - 0.081312 * img[:,:,2] + 128
    return Y, Cb, Cr

def ycbcr_to_rgb(Y, Cb, Cr):
    R = Y + 1.402   * (Cr - 128)
    G = Y - 0.344136* (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772   * (Cb - 128)
    rgb = np.stack((R, G, B), axis=-1)
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)


# Step 2: Chroma Subsampling (4:2:0)

def subsample(channel):
    """
    For 4:2:0 subsampling: take every 2nd row and every 2nd column.
    """
    return channel[::2, ::2]

def upsample(channel, shape):
    """
    Upsamples a subsampled channel using nearest-neighbor to the target shape.
    """
    upsampled = np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)
    return upsampled[:shape[0], :shape[1]]


# Step 3: Splitting into blocks

def block_split(channel, block_size=8):
    """
    Splits a 2D channel into non-overlapping block_size x block_size blocks.
    Pads the channel if necessary.
    Returns a list of blocks and the padded shape.
    """
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    h_new, w_new = padded.shape
    blocks = []
    for i in range(0, h_new, block_size):
        for j in range(0, w_new, block_size):
            blocks.append(padded[i:i+block_size, j:j+block_size])
    return blocks, padded.shape

def block_merge(blocks, shape, block_size=8):
    """
    Reconstructs a 2D channel from 8x8 blocks.
    """
    h, w = shape
    merged = np.zeros((h, w))
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            merged[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return merged[:h, :w]


# Step 4: DCT and Inverse DCT

def dct2(block):
    """
    Applies 2D DCT (type-II) to an 8x8 block.
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """
    Applies inverse 2D DCT (type-III) to an 8x8 block.
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# Step 5: Quantization

# Standard JPEG quantization matrices for luminance and chrominance:
QY = np.array([[16,11,10,16,24,40,51,61],
               [12,12,14,19,26,58,60,55],
               [14,13,16,24,40,57,69,56],
               [14,17,22,29,51,87,80,62],
               [18,22,37,56,68,109,103,77],
               [24,35,55,64,81,104,113,92],
               [49,64,78,87,103,121,120,101],
               [72,92,95,98,112,100,103,99]])
QC = np.array([[17,18,24,47,99,99,99,99],
               [18,21,26,66,99,99,99,99],
               [24,26,56,99,99,99,99,99],
               [47,66,99,99,99,99,99,99],
               [99,99,99,99,99,99,99,99],
               [99,99,99,99,99,99,99,99],
               [99,99,99,99,99,99,99,99],
               [99,99,99,99,99,99,99,99]])

def scale_quant_matrix(Q, quality):
    """
    Scales the quantization matrix Q based on the quality factor.
    """
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    Q_scaled = np.floor((Q * scale + 50) / 100)
    Q_scaled[Q_scaled == 0] = 1  # Prevent division by zero.
    return Q_scaled

def quantize(block, Q):
    """
    Quantizes a DCT block by dividing element-wise by Q and rounding.
    """
    return np.round(block / Q).astype(np.int32)

def dequantize(block, Q):
    """
    Dequantizes a block by multiplying element-wise by Q.
    """
    return (block * Q).astype(np.float32)


# Step 6: Zigzag Order and Entropy Encoding

def zigzag_indices(n=8):
    """
    Computes the zigzag scan order indices for an n x n block.
    """
    indices = np.empty((n*n, 2), dtype=int)
    index = -1
    for s in range(0, 2*n - 1):
        if s % 2 == 0:
            x = min(s, n - 1)
            y = s - x
            while x >= 0 and y < n:
                index += 1
                indices[index] = [x, y]
                x -= 1
                y += 1
        else:
            y = min(s, n - 1)
            x = s - y
            while y >= 0 and x < n:
                index += 1
                indices[index] = [x, y]
                x += 1
                y -= 1
    return indices

# Precompute zigzag order indices for an 8x8 block.
zigzag_ind = zigzag_indices(8)

def zigzag_order(block):
    """
    Returns the elements of an 8x8 block in zigzag order.
    """
    return np.array([block[i, j] for i, j in zigzag_ind])

def inverse_zigzag_order(arr):
    """
    Reconstructs an 8x8 block from a 1D array in zigzag order.
    """
    block = np.zeros((8,8))
    for index, (i, j) in enumerate(zigzag_ind):
        block[i, j] = arr[index]
    return block

def run_length_encode(arr):
    encoded = []
    count = 0
    for val in arr:
        if val == 0:
            count += 1
        else:
            encoded.append((count, val))
            count = 0
    encoded.append((0, 0))  # End-of-block marker
    return encoded

def run_length_decode(encoded):
    decoded = []
    for count, val in encoded:
        if (count, val) == (0, 0):  # End-of-block marker
            break
        decoded.extend([0] * count)
        decoded.append(val)
    while len(decoded) < 64:  # Ensure 64 coefficients per block
        decoded.append(0)
    return np.array(decoded)


# Processing Function for Each Channel

def process_channel(blocks, Q):
    processed_blocks = []
    for block in blocks:
        block = block - 128          
        dct_block = dct2(block)       
        quant_block = quantize(dct_block, Q)
        zz = zigzag_order(quant_block)
        rle = run_length_encode(zz)
        processed_blocks.append(rle)
    return processed_blocks

def inverse_process_channel(processed_blocks, Q, shape):
    blocks = []
    for rle in processed_blocks:
        zz = run_length_decode(rle)
        quant_block = inverse_zigzag_order(zz)
        dequant_block = dequantize(quant_block, Q)
        block = idct2(dequant_block)
        block = block + 128
        block = np.clip(block, 0, 255)
        blocks.append(block)
    channel = block_merge(blocks, shape)
    return channel

def compress_image(image_path, quality=50, 
                   output_compressed="compressed.bin", 
                   output_jpeg="output_compressed.jpg"):

    # Load the PNG image and convert it to RGB.
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    # Convert RGB to YCbCr.
    Y, Cb, Cr = rgb_to_ycbcr(img_np)
    
    # Subsample chrominance channels (4:2:0).
    Cb_sub = subsample(Cb)
    Cr_sub = subsample(Cr)
    
    # Split each channel into 8x8 blocks.
    Y_blocks, Y_shape = block_split(Y, 8)
    Cb_blocks, Cb_shape = block_split(Cb_sub, 8)
    Cr_blocks, Cr_shape = block_split(Cr_sub, 8)
    
    # Scale the quantization matrices.
    QY_scaled = scale_quant_matrix(QY, quality)
    QC_scaled = scale_quant_matrix(QC, quality)
    
    # Process each channel.
    Y_processed  = process_channel(Y_blocks, QY_scaled)
    Cb_processed = process_channel(Cb_blocks, QC_scaled)
    Cr_processed = process_channel(Cr_blocks, QC_scaled)
    
    # Package the compressed data and header information into a dictionary.
    compressed_data = {
        "Y": Y_processed,
        "Cb": Cb_processed,
        "Cr": Cr_processed,
        "Y_shape": Y_shape,
        "Cb_shape": Cb_shape,
        "Cr_shape": Cr_shape,
        "image_shape": img_np.shape,
        "quality": quality
    }
    
    # Save the compressed data to a binary file.
    with open(output_compressed, "wb") as f:
        pickle.dump(compressed_data, f)
    print("Compression complete. Compressed data saved to", output_compressed)
    

    QY_scaled = scale_quant_matrix(QY, quality)
    QC_scaled = scale_quant_matrix(QC, quality)
    
    # Reconstruct each channel from the compressed data.
    Y_channel  = inverse_process_channel(compressed_data["Y"], QY_scaled, Y_shape)
    Cb_channel = inverse_process_channel(compressed_data["Cb"], QC_scaled, Cb_shape)
    Cr_channel = inverse_process_channel(compressed_data["Cr"], QC_scaled, Cr_shape)
    
    # Upsample the chrominance channels to the original image dimensions.
    orig_shape = compressed_data["image_shape"]
    Cb_upsampled = upsample(Cb_channel, (orig_shape[0], orig_shape[1]))
    Cr_upsampled = upsample(Cr_channel, (orig_shape[0], orig_shape[1]))
    
    # Convert YCbCr back to RGB.
    rgb = ycbcr_to_rgb(Y_channel, Cb_upsampled, Cr_upsampled)
    img_out = Image.fromarray(rgb)
    
    # Save the reconstructed image as a standard JPEG.
    img_out.save(output_jpeg, "JPEG")
    print("Compressed JPEG image saved as", output_jpeg)

def main():
    input_image = "input_image.png"  
    quality = 50                    
    
    print("Starting compression...")
    compress_image(input_image, quality, 
                   output_compressed="compressed.bin", 
                   output_jpeg="output_image.jpg")

if __name__ == "__main__":
    main()
