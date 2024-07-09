import argparse
import os

import torch
from adversarial_image_library import add_noise, target_model, preprocess_image


def test_noise_addition(test_image_path: str, target_class: int):
    """Applies noise addition pipeline and checks the model predictions"""
    adversarial_image = add_noise(test_image_path, target_class)
    adversarial_image = preprocess_image(adversarial_image).unsqueeze(0)
    predicted_class = target_model(adversarial_image).max(1)[1]
    assert int(predicted_class.item()) == target_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adversarial noise for an image.')
    parser.add_argument('--test_data_path', type=str, default='data',
                        help='Path to the directory containing the images for test.')
    parser.add_argument('--target_class', type=int, default=355,
                        help='Target class for the adversarial attack on ResNet-50.')

    args = parser.parse_args()

    correct_answers = 0
    wrong_answered_files = []
    for image in os.listdir(args.test_data_path):
        test_image_path = os.path.join(args.test_data_path, image)
        try:
            test_noise_addition(test_image_path, args.target_class)
            correct_answers += 1
        except AssertionError:
            wrong_answered_files.append(test_image_path)
    print(f'Correct target answers: {correct_answers}/{len(os.listdir(args.test_data_path))}')
    print(f'Wrong answered files: {wrong_answered_files}')
