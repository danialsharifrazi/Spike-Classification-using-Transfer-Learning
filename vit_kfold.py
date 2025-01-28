import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import datetime
import print_results
from extract_spikes import load_spikes
from keras.callbacks import CSVLogger
from keras import layers
import tensorflow as tf
from keras.models import Model




dpi=0
data,labels=load_spikes(dpi)
print(data.shape,labels.shape)

data = np.expand_dims(data, axis=3)
data=data.astype('int8')
data=np.resize(data, (data.shape[0], 224, 224, 1))
data=data.astype('int8')
from keras.utils import to_categorical
labels=to_categorical(labels)


def create_model(input_shape, num_classes, patch_size, projection_dim, num_heads, mlp_dim, transformer_layers):
    inputs = layers.Input(shape=input_shape)
    
    patch_size = patch_size
    patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    patches = layers.Reshape((-1, projection_dim))(patches) 

    num_patches = patches.shape[1]
    position_embedding = tf.Variable(tf.random.normal([1, num_patches, projection_dim]))
    x = patches + position_embedding

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([x, attention_output])  

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp_output = layers.Dense(mlp_dim, activation="relu")(x3)
        mlp_output = layers.Dense(projection_dim)(mlp_output)
        x = layers.Add()([x2, mlp_output])  

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model


input_shape = (224, 224, 1)
num_classes = 3
patch_size = 4
projection_dim = 64
num_heads = 4
mlp_dim = 128
transformer_layers = 2
model_name='ViT'


fold_number=1
lst_acc,lst_precision,lst_recall,lst_f1,lst_matrix,lst_times=[],[],[],[],[],[]
kf=KFold(n_splits=10,shuffle=True)
for train,test in kf.split(data,labels):

    callback=CSVLogger(f'./results/vit_model/{dpi}dpi/{model_name}_logger_{fold_number}.log')
    
    x_train=data[train]
    x_test=data[test]
    y_train=labels[train]
    y_test=labels[test]


    model=create_model(input_shape, num_classes, patch_size, projection_dim, num_heads, mlp_dim, transformer_layers)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    strat_time=datetime.datetime.now()
    model.fit(x_train, y_train, batch_size=8, epochs=10, validation_split=0.1, callbacks=[callback])
    end_time=datetime.datetime.now()
    training_time=end_time-strat_time

    predicts=model.predict(x_test, batch_size=8)
    predicts=predicts.argmax(axis=1)
    actuals=y_test.argmax(axis=1)

    lst_acc.append(accuracy_score(actuals,predicts))
    lst_precision.append(precision_score(actuals,predicts,average='macro'))
    lst_recall.append(recall_score(actuals,predicts,average='macro'))
    lst_f1.append(f1_score(actuals,predicts,average='macro'))
    lst_matrix.append(confusion_matrix(actuals,predicts))
    lst_times.append(training_time)

    fold_number+=1
    
print_results.results_vit(lst_acc,lst_precision,lst_recall,lst_f1,lst_matrix,lst_times,model_name,dpi)

