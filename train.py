from homework.train_classification import train as train_classifier
from homework.train_detection import train as train_detector

if __name__ == '__main__':
    # train_classifier(lr=2e-3)

    train_detector(lr=2e-2, batch_size = 128, num_epoch=20)
    train_detector(lr=1e-2, batch_size = 128, num_epoch=20)
    train_detector(lr=2e-3, batch_size = 128, num_epoch=20)
