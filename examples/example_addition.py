from adversarial_image_library import add_noise, save_image
import os

if __name__ == '__main__':
    image_dir_path = 'data'
    save_dir_path = 'adversarial_data'
    os.makedirs(save_dir_path, exist_ok=True)

    os.path.join(save_dir_path)

    image_file = 'panda.jpg'
    image_path = os.path.join(image_dir_path, image_file)
    save_path = os.path.join(save_dir_path, image_file)
    target_class = 355  # Example target class: llama

    # add noise to image
    adversarial_image = add_noise(image_path, target_class)

    # show and save resulting image
    adversarial_image.show()
    save_image(adversarial_image, save_path)
    print(f"Adversarial image saved as '{save_dir_path}/{image_file}'")
