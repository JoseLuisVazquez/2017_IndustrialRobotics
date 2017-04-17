import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

train, test, _ = imdb.load_data(path ='imdb.pkl', n_words = 10000, valid_portion =0.1)

trainX, trainY = train
textX, testY = test

trainX = pad_sequences(trainX, maxlen = 100, value = 0.)
testX = pad_sequences(testX, maxlen =100, value = 0. )

#converting labels to binary vectors

trainY = to_categorical(trainY, nbclasses = 2)
testY = to_categorical(testY, nbclasses = 2)

#Network building
net =tflearn.input_data([[none, 100]])
net = tflearn.embedding (net, input_dim =10000, output_dim =128)
net = tflearn.lsstm(net,128, dropout = 0.8)
net =tflearn.fully_connected(net,2, activation = 'softmax')
net =tflearn.regression(net, optimizer = 'adam', learning_rate = 0.0001, loss = 'categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose = 0)
model.fit(trainX, trainY, validation_set(testX,testY), show_metric = True, batch_size = 32)