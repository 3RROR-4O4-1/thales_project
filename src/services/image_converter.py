import argparse
from PIL import Image

def convert_png_to_jpg(input_file, output_file):
    # Open the PNG image
    png_image = Image.open(input_file)
    
    # Convert PNG to JPG (remove transparency if any)
    jpg_image = png_image.convert('RGB')
    
    # Save the image as JPG
    jpg_image.save(output_file, 'JPEG')
    print(f"Conversion successful! Saved as {output_file}")

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Convert PNG to JPG")
    
    # Add arguments for input and output filenames
    parser.add_argument('input_file', type=str, help="Path to the input PNG file")
    parser.add_argument('output_file', type=str, help="Path to save the output JPG file")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the conversion function
    convert_png_to_jpg(args.input_file, args.output_file)




## python image_converter.py input_image.png output_image.jpg
