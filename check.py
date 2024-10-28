from homework.models import Classifier, calculate_model_size_mb

if __name__ == '__main__':
  model = Classifier()
  model_size = calculate_model_size_mb(model)
  print(f"Model size: {model_size:.2f} MB")
  print(model)
