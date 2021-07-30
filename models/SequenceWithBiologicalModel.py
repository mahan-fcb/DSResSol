from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import concatenate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import sklearn.metrics
from tensorflow.keras.optimizers import Adam

from models.DsResSolModel import DsResSolModel
from lib.construct import residual_block


class SequenceWithBiologicalModel(DsResSolModel):
    def __init__(self, util):
        super().__init__(util)

    def train(self):
        prepared_data = self.prepare()
        train_pad = prepared_data['train_pad']
        test_pad = prepared_data['test_pad']
        test_pad1 = prepared_data['test_pad']
        y_train = prepared_data['y_train']
        y_train_le = prepared_data['y_train_le']
        y_test = prepared_data['y_test']
        y_test1 = prepared_data['y_test1']
        skf = prepared_data['skf']
        t = prepared_data['t']
        b = prepared_data['b']
        t1 = prepared_data['t1']
        b1 = prepared_data['b1']
        train_ohe = prepared_data['train_ohe']
        X = prepared_data['X']
        X1 = prepared_data['X1']
        X2 = prepared_data['X2']

        k = 0
        epochs = self.util.config['model_config']['sequence_and_biological']['epochs'].get(
            int)
        batch_size = self.util.config['model_config']['sequence_and_biological']['batch_size'].get(
            int)

        for train_index, test_index in skf.split(train_ohe, y_train_le):
            X_train, X_test = train_pad[train_index], train_pad[test_index]
            X_train2, X_test2 = X[train_index], X[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]

            optimizer = Adam(learning_rate=self.util.config['model_config']['sequence_and_biological']['optimizer']['adam']['learning_rate'].get(
            ), decay=1e-7)
            x_input = Input(
                shape=(self.util.config['model_config']['x_input']['shape'].get(int),))
            input2 = Input(shape=(self.util.config['model_config']['sequence_and_biological']['input2']['shape'].get(),))
            conv = Embedding(21, 50, input_length=self.util.config['model_config']['x_input']['shape'].get(int))(
                x_input)  # initial conv
            res1 = residual_block(conv, 32, 1, 2)
            x = MaxPooling1D(3)(res1)
            res2 = residual_block(conv, 32, 2, 2)
            x1 = MaxPooling1D(3)(res2)
            res3 = residual_block(conv, 32, 3, 2)
            x2 = MaxPooling1D(3)(res3)
            res4 = residual_block(conv, 32, 4, 2)
            x3 = MaxPooling1D(3)(res4)
            res5 = residual_block(conv, 32, 5, 2)
            x4 = MaxPooling1D(3)(res5)
            res6 = residual_block(conv, 32, 6, 2)
            x5 = MaxPooling1D(3)(res6)
            res7 = residual_block(conv, 32, 7, 2)
            x6 = MaxPooling1D(3)(res7)
            res8 = residual_block(conv, 32, 8, 2)
            x7 = MaxPooling1D(3)(res8)
            res9 = residual_block(conv, 32, 9, 2)
            x8 = MaxPooling1D(3)(res9)
            x_final1 = concatenate([x8, x7, x6, x5, x4, x3, x2, x1, x])
            x9 = Conv1D(32, 11, padding='same', activation='relu')(x_final1)
            x9 = MaxPooling1D(3)(x9)
            x10 = Conv1D(32, 13, padding='same', activation='relu')(x_final1)
            x10 = MaxPooling1D(3)(x10)
            x11 = Conv1D(32, 15, padding='same', activation='relu')(x_final1)
            x11 = MaxPooling1D(3)(x11)
            x_final = concatenate([x11, x10, x9])
            x_final = Flatten()(x_final)
            x_final = concatenate([x_final, input2])
            x_final = Dense(256, activation='relu')(x_final)
            x_output = Dense(2, activation='softmax',
                             kernel_regularizer=l2(0.0001))(x_final)
            model2 = Model(inputs=[x_input, input2], outputs=x_output)
            model2.compile(optimizer=optimizer,
                           loss='categorical_crossentropy', metrics=['accuracy'])
            es = EarlyStopping(monitor='val_loss', patience=4, verbose=1)

            his = model2.fit([X_train, X_train2], Y_train, batch_size=batch_size, callbacks=[
                             es],  epochs=epochs, validation_data=([X_test, X_test2], Y_test), verbose=1,  shuffle=True)

            y_pred = model2.predict([test_pad, X1]).round().astype(int)
            y_pred1 = model2.predict([test_pad1, X2]).round().astype(int)
            t[k] = sklearn.metrics.f1_score(y_test, y_pred, average='weighted')
            b[k] = accuracy_score(y_test, y_pred)
            t1[k] = sklearn.metrics.f1_score(
                y_test1, y_pred1, average='weighted')
            b1[k] = accuracy_score(y_test1, y_pred1)
            print(t[k], b[k], t1[k], b1[k])
            model2.save('model_new_bot1'+str(k)+".h5")
            k += 1
