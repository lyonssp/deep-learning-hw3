from homework.train_classification import train as train_classifier
from homework.train_detection import train as train_detector

if __name__ == '__main__':
    # train_classifier(lr=2e-3)

    train_detector(lr=2e-3, batch_size = 128, num_epoch=20, alpha = 1, beta = 0.75)
    train_detector(lr=2e-3, batch_size = 128, num_epoch=20, alpha = 1, beta = 0.5)
    train_detector(lr=2e-3, batch_size = 128, num_epoch=20, alpha = 1, beta = 0.25)
