import argparse

from adversarial_image_library import add_noise, save_image
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adversarial noise for an image.')
    parser.add_argument('--image_dir_path', type=str, default='data', help='Path to the directory containing the image.')
    parser.add_argument('--save_dir_path', type=str, default='adversarial_data', help='Path to the directory to save the adversarial image.')
    parser.add_argument('--image_file', type=str, default='panda.jpg', help='Name of the image file.')
    parser.add_argument('--target_class', type=int, default=355, help='Target class for the adversarial attack on ResNet-50.')

    args = parser.parse_args()

    os.path.join(args.save_dir_path)
    image_path = os.path.join(args.image_dir_path, args.image_file)
    save_path = os.path.join(args.save_dir_path, args.image_file)
    target_class = 355  # Example target class: llama

    # add noise to image
    adversarial_image = add_noise(image_path, target_class)

    # show and save resulting image
    adversarial_image.show()
    save_image(adversarial_image, save_path)
    print(f"Adversarial image saved as '{args.save_dir_path}/{args.image_file}'")
