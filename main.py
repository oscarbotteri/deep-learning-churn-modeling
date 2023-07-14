import os
from keras.saving import load_model
from sklearn.preprocessing import StandardScaler
from model import load


def get_model(existing: bool):
    if not existing:
        return load()

    model = load_model("model.keras")

    print("Loaded model from disk")

    return model


def main():
    exists = os.path.exists("model.keras")

    if not exists:
        print("No previously model found, generating new one")

    model = get_model(exists)
    sc_x = StandardScaler()

    # TODO: Read values from CLI
    prediction = model.predict(
        sc_x.fit_transform([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
    )

    print("Susceptibility to leave {:.1%}".format(prediction[0][0]))


main()
