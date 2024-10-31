from homework.models import Classifier, Detector, calculate_model_size_mb
import torch

if __name__ == '__main__':
  classifier = Classifier()
  classifier_size = calculate_model_size_mb(classifier)
  print(f"Classifier model size: {classifier_size:.2f} MB")
  print(classifier)

  detector = Detector()
  detector_size = calculate_model_size_mb(detector)
  print(f"Detector model size: {detector_size:.2f} MB")
  print(detector)

  x = torch.randn(128, 3, 96, 128)
  y = detector.forward(x)
  print(f"logits output shape: {y[0].shape}")
  print(f"depth output shape: {y[1].shape}")
