import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load():
    dataset = pd.read_csv("Churn_Modelling.csv")

    x = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # Encode labels
    country_label_encoder = LabelEncoder()
    gender_label_encoder = LabelEncoder()

    x[:, 1] = country_label_encoder.fit_transform(x[:, 1])
    x[:, 2] = gender_label_encoder.fit_transform(x[:, 2])

    # Transform categorical features
    transformer = ColumnTransformer(
        transformers=[("Churn_Modelling", OneHotEncoder(categories="auto"), [1])],
        remainder="passthrough",
    )

    x = transformer.fit_transform(x)
    x = x[:, 1:]

    # Split dataset in training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    # Scale variables
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    # Initialize ANN and set layers
    model = Sequential()
    model.add(
        Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11)
    )
    model.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

    # Compile and fit
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=10, epochs=100)

    # Predict values using testing dataset
    y_pred = model.predict(x_test)
    y_pred = y_pred > 0.5

    # Make a confusion matrix and print result
    matrix = confusion_matrix(y_test, y_pred)
    print("Accuracy", (matrix[0][0] + matrix[1][1]) / matrix.sum())

    # Save model to disk
    model.save("model.keras")
    print("The model has been saved to disk")

    return model
