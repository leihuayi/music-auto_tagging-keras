from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense
import json
import audio_processor as ap
import numpy as np

# Model parameters
nb_classes = 10
nb_epoch = 40
batch_size = 10

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


def retrain_model(initial_model):
    print("Retraining the model ...")

    # Eliminate last layer
    pop_layer(initial_model)

    # Add new Dense layer
    last = initial_model.get_layer('dropout_5')
    preds = (Dense(nb_classes, activation='sigmoid', name='preds'))(last.output)
    model = Model(initial_model.input, preds)

    # freeze weights except last 3 layers
    for layer in model.layers[:3]:
        layer.trainable = False

    # stochastic gradient descent
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    X, Y = load_data()

    print("Fitting the model ...")
    model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_split=0.2, shuffle=True)
    return model 


def load_data():
    tags = ['rock', 'pop', 'rnb', 'Hip-Hop', 'rap', 'electronic', 'sad']

    dataPath = "../Tests/Data/"
    X_list=[]
    Y_list=[]
    with open("../statistics/genres.json", "r") as read_file:
        data = json.load(read_file)
        for s, label in data.items():
            X_list.append(dataPath + s+".mp3")
            Y_list.append(label)
    
    n = len(X_list)
    X= np.zeros((0, 1, 96, 1366))
    Y= np.zeros((n,nb_classes))
    print("Computing melgrams for training data...")
    for i in range(n):
        # Load input image
        melgram = ap.compute_melgram(X_list[i])
        X = np.concatenate((X, melgram), axis=0)
        ind = tags.index(Y_list[i])
        Y[i][ind] = 1
    
    # Shuffle the training data
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    return X, Y
