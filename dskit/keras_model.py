## Import Keras deep neural network library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l1, l2


## Function to create model, required for KerasClassifier
def create_model(nlayers=1, nneurons=100, dropout_rate=0.0,
                 l2_norm=1e-3, activation='relu', kernel_initializer='lecun_normal',
                 optimizer='adam', input_dim=None, metric='accuracy'):

    # create model
    model = Sequential()

    # set input layer
    model.add(Dense(nneurons,
                        input_dim=input_dim,
                        kernel_initializer=kernel_initializer,
                        activation=activation,
                        kernel_regularizer=l2(l2_norm)))
    model.add(Dropout(dropout_rate))

    # indicate the number of layers
    for layer in range(nlayers-1):
        model.add(Dense(nneurons, 
                        input_dim=input_dim,
                        kernel_initializer=kernel_initializer,
                        activation=activation, 
                        kernel_regularizer=l2(l2_norm)))
        model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(1, # 2
                    kernel_initializer=kernel_initializer,
                    activation='sigmoid',  # activation='softmax',
                    kernel_regularizer=l2(l2_norm)))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), #optimizer=optimizer,
                  metrics=[metric])

    return model
