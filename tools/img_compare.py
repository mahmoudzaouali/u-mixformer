from PIL import Image, ImageChops
import numpy as np


def convert_to_binary(image, threshold=128):
    """Convert an image to a binary (black and white) image."""
    # Convert the image to grayscale
    gray_image = image.convert("L")
    # Apply threshold
    binary_image = gray_image.point(lambda x: 255 if x > threshold else 0, "1")
    return binary_image


def compare_images(image1, image2):
    """Compare two binary images and show the difference."""
    # Ensure the images are the same size
    if image1.size == image2.size:
        diff = ImageChops.difference(image1, image2)
        result = Image.new("RGB", image1.size, (0, 0, 0))
        for x in range(diff.size[0]):
            for y in range(diff.size[1]):
                if diff.getpixel((x, y)) != 0:
                    result.putpixel((x, y), (255, 0, 0))
                else:
                    g = image1.getpixel((x, y))
                    result.putpixel((x, y), (g, g, g))
        result.show()
    else:
        print("Images have different sizes and cannot be compared.")


# Load the original binary image
img_dir = "./data/img_dir/visualize/"
img_name_1 = "image_3731.jpg"
img_name_2 = "output_3731.jpg"
img_name_3 = "image_3731.png"
img_full_name_1 = img_dir + img_name_1  # original
img_full_name_2 = img_dir + img_name_2  # prediction
img_full_name_3 = img_dir + img_name_3  # ground truth

img1 = Image.open(img_full_name_1)
img2 = Image.open(img_full_name_2)
img3 = Image.open(img_full_name_3)

# Convert the second image to binary
img2 = convert_to_binary(img2, threshold=142)  # For the prediction image 135 threshold
# ia s good choice
img2.show()
img3.show()

img1 = convert_to_binary(img1, threshold=128)  # Adjust the threshold as needed
# Compare the two binary images
compare_images(img2, img3)
