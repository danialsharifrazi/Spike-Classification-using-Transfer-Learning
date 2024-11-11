import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import datetime
import print_results
from extract_spikes import load_spikes
from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.applications import EfficientNetB7 as pre_trained_net


def create_model():
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D

    base_model = pre_trained_net(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    model_output = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=model_output)

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return model



# Load spikes sequence
dpi=0
data,labels=load_spikes(dpi)
print(data.shape,labels.shape)

# Preprocessing
data=data.astype('uint8')
data = np.repeat(data[..., np.newaxis], 3, -1)
data=np.array([np.resize(item,(224, 224,3)) for item in data])
data=data.astype('uint8')
labels=to_categorical(labels)


model_name='EfficientNet'
fold_number=1
lst_acc,lst_precision,lst_recall,lst_f1,lst_matrix,lst_times=[],[],[],[],[],[]
kf=KFold(n_splits=10, shuffle=True)
for train,test in kf.split(data, labels):

    callback=CSVLogger(f'./{model_name}_logger_{fold_number}.log')
    model=create_model()
    
    x_train=data[train]
    x_test=data[test]
    y_train=labels[train]
    y_test=labels[test]


    model.fit(x_train,y_train,batch_size=128,epochs=20)
    for layer in model.layers[:-50]:
        layer.trainable = False
    for layer in model.layers[-50:]:
        layer.trainable = True
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    strat_time=datetime.datetime.now()
    model.fit(x_train,y_train,batch_size=128,epochs=20,validation_split=0.1,callbacks=[callback])
    end_time=datetime.datetime.now()
    training_time=end_time-strat_time

    predicts=model.predict(x_test)
    predicts=predicts.argmax(axis=1)
    actuals=y_test.argmax(axis=1)

    lst_acc.append(accuracy_score(actuals,predicts))
    lst_precision.append(precision_score(actuals,predicts,average='macro'))
    lst_recall.append(recall_score(actuals,predicts,average='macro'))
    lst_f1.append(f1_score(actuals,predicts,average='macro'))
    lst_matrix.append(confusion_matrix(actuals,predicts))
    lst_times.append(training_time)

    fold_number+=1
    
print_results.results(lst_acc,lst_precision,lst_recall,lst_f1,lst_matrix,lst_times,model_name,dpi)

