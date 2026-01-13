import tensorflow as tf
from preprocess import get_data_generators
from model import build_model

TRAIN_DIR = "data/raw/train"
VAL_DIR = "data/raw/val"
MODEL_PATH = "model/tomato_model.h5"

def train():
    train_gen, val_gen = get_data_generators(TRAIN_DIR, VAL_DIR)

    num_classes = train_gen.num_classes
    model = build_model(num_classes)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    model.save(MODEL_PATH)
    print("✅ Model saved at", MODEL_PATH)

if __name__ == "__main__":
    train()
