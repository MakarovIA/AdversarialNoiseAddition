import torchvision.models as models

# Pre-trained model under attack
target_model = models.resnet50(pretrained=True)
target_model.eval()

# preprocess params
preprocess_mean = [0.485, 0.456, 0.406]
preprocess_std = [0.229, 0.224, 0.225]