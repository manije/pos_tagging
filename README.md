# pos_tagging

**<div dir=rtl><font size=4> می خواهیم یک  POS با استفاده از Keras و یک لایه LSTM Bidirectional ایجاد کنیم.</div>**


# <div dir='rtl'>  **خواندن-اطلاعات** </div>

    <div> change from github </div>

## <div dir='rtl'> **تغییرات-روی-داده-ورودی** </div>
<div dir='rtl'><font size=3>  پایگاه داده مورد استفاده : UPC  </div>
<div dir='rtl'> باید ساختار داده ها را تغییر دهیم:    </div>



<div dir='rtl'>1.   جملات را در پایگاه داده مشخص کنیم </div>
<div dir='rtl'>2.    کلمات و برچسب ها را از هم جدا کنیم</div>


<div dir='rtl'>یعنی برای هر جمله یک آرایه خواهیم داشت که هر کدام از المان های آن یک کلمه از آن جمله است و متناظر با آن یک آرایه برای برچسب داریم که هر کدام بصورت جدا هر کلمه برچسب آن را قرار می دهیم. </div>




```python
from google.colab import drive
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    


```python
import numpy as np
doc = []
doc_tag = []
sen =[]
tag =[]
with open('/content/drive/My Drive/dataset/NLP/UPC.txt','r') as f:
    for line in f:      
      # print(line)
      words = line.split()
      if (len(words)==2):
        sen.append(words[0])
        tag.append(words[1])
      elif(len(sen)!=0):
        doc.append(np.array(sen))
        doc_tag.append(np.array(tag))
        sen = []
        tag =[]
sentences_2 = doc[int(len(doc)*0.50):int(len(doc)* 0.6)]
sentence_tags_2 = doc_tag[int(len(doc_tag)*0.50):int(len(doc_tag)* 0.6)]
```


```python
print(sentences_2[0:2])
print(sentence_tags_2[0:2])
print(len(sentences_2))
```

    [array(['وی', 'گفت', ':', 'برخی', 'از', 'این', 'مردم', '،', 'به', 'بیش',
           'از', '۶۲', 'زبان', 'زنده', 'دنیا', 'به', 'من', 'حمله',
           'می\u200cکنند', 'و', 'می\u200cگویند', 'به', 'خانه\u200cات',
           'برگرد', '.'], dtype='<U8'), array(['اما', 'من', 'به', 'این', 'مخالفتها', 'عادت', 'دارم', 'و', 'در',
           'طول', '۳۱', 'سالی', 'که', 'در', 'کار', 'سیاسی', 'بوده\u200cام',
           '،', 'یاد', 'گرفته\u200cام', 'که', 'در', 'بسیاری', 'از', 'موارد',
           'هدف', 'مخالفان', '،', 'ارزیابی', 'و', 'دیدن', 'واکنش', 'من',
           'است', '.'], dtype='<U8')]
    [array(['PRO', 'V_PA', 'DELM', 'PRO', 'P', 'DET', 'N_SING', 'DELM', 'P',
           'ADJ', 'P', 'NUM', 'N_SING', 'ADJ', 'N_SING', 'P', 'PRO', 'N_SING',
           'V_PRS', 'CON', 'V_PRS', 'P', 'N_SING', 'V_IMP', 'DELM'],
          dtype='<U6'), array(['CON', 'PRO', 'P', 'DET', 'N_PL', 'N_SING', 'V_PRS', 'CON', 'P',
           'N_SING', 'NUM', 'N_SING', 'CON', 'P', 'N_SING', 'ADJ', 'V_PP',
           'DELM', 'N_SING', 'V_PP', 'CON', 'P', 'ADJ', 'P', 'N_PL', 'N_SING',
           'N_PL', 'DELM', 'N_SING', 'CON', 'N_SING', 'N_SING', 'PRO',
           'V_PRS', 'DELM'], dtype='<U6')]
    8935
    

## <div dir='rtl'> **تقسیم داده:** </div>

<div dir='rtl'><font size=3>  قبل از آموزش یک مدل ، باید داده ها را به دو دسته آموزش و آزمایش تقسیم کنیم. بدین منظور از عملکرد train_test_split از Scikit-Learn استفاده میکنیم: </div>


```python
from sklearn.model_selection import train_test_split
(train_sentences_2, 
 test_sentences_2, 
 train_tags_2, 
 test_tags_2) = train_test_split(sentences_2, sentence_tags_2, test_size=0.4)
```


```python
print(test_sentences_2[0:2])
```

    [array(['#', 'توضیحی', 'درباره', 'یک', 'ستون', 'جدید', '#', 'ستون', 'مجلس',
           'و', 'نطق\u200cهای', 'نمایندگانش', '،', 'مجلس', 'و', 'مصوباتش',
           'ستون', 'تازه\u200cای', 'در', 'اطلاعات', 'بین\u200cالمللی', 'است',
           'که', 'اولین', 'قسمت', 'آن', 'را', 'روز', 'جمعه', 'چهارم',
           'شهریور', 'خواندید', '.'], dtype='<U10'), array(['این', 'خبر', 'در', 'نشست', 'شورای', 'بهداشت', 'استان', 'تهران',
           'که', 'روز', 'شنبه', 'در', 'محل', 'استانداری', 'تشکیل', 'شد', '،',
           'اعلام', 'گردید', '.'], dtype='<U9')]
    

## <div dir='rtl'> **تغییرات مربوط به داده های برای کار با kerass**: </div>
<div dir='rtl'><font size=3.5> Keras  باید با اعداد کار کند ، نه با کلمات (یا برچسب ها).به همین منظور به هر کلمه (و برچسب) یک عدد صحیح منحصر به فرد اختصاص می دهیم. </div>
<div dir = 'rtl'>
  محاسبه مجموعه ای از کلمات منحصر به فرد (و برچسب ها)  آن را در یک لیست  قرار می دهیم.
<div dir = 'rtl'> همچنین یک مقدار ویژه برای پر کردن کلمات ناشناخته اضافه خواهیم کرد (OOV - Out Of Vocabulary).
 



```python
words_2, tags_2 = set([]), set([])
 
for s in train_sentences_2:
    for w in s:
        words_2.add(w.lower())
 
for ts in train_tags_2:
    for t in ts:
        tags_2.add(t)
        
word2index_2 = {w: i + 2 for i, w in enumerate(list(words_2))}
word2index_2['-PAD-'] = 0  # The special value used for padding
word2index_2['-OOV-'] = 1  # The special value used for OOVs
 
tag2index_2 = {t: i + 1 for i, t in enumerate(list(tags_2))}
tag2index_2['-PAD-'] = 0  # The special value used to padding
```


```python
from keras.preprocessing.sequence import pad_sequences
train_sentences_X_2, test_sentences_X_2, train_tags_y_2, test_tags_y_2 = [], [], [], []
 
for s in train_sentences_2:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index_2[w.lower()])
        except KeyError:
            s_int.append(word2index_2['-OOV-'])
    train_sentences_X_2.append(s_int)
 
for s in test_sentences_2:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index_2[w.lower()])
        except KeyError:
            s_int.append(word2index_2['-OOV-'])
    test_sentences_X_2.append(s_int)
 
for s in train_tags_2:
    train_tags_y_2.append([tag2index_2[t] for t in s])
 
for s in test_tags_2:
    test_tags_y_2.append([tag2index_2[t] for t in s])


MAX_LENGTH_2 = len(max(train_sentences_X_2, key=len))

print(train_sentences_X_2[0])
print(test_sentences_X_2[0])
print(train_tags_y_2[0])
print(test_tags_y_2[0])

train_sentences_X_2 = pad_sequences(train_sentences_X_2, maxlen=MAX_LENGTH_2, padding='post')
test_sentences_X_2 = pad_sequences(test_sentences_X_2, maxlen=MAX_LENGTH_2, padding='post')
train_tags_y_2 = pad_sequences(train_tags_y_2, maxlen=MAX_LENGTH_2, padding='post')
test_tags_y_2 = pad_sequences(test_tags_y_2, maxlen=MAX_LENGTH_2, padding='post')
 
```

    [4924, 11027, 698, 748, 6354, 4462, 6818, 8890, 9426, 2633, 7228, 4726, 10523, 4595, 2633, 11979, 11405, 697, 2633, 15615, 4924, 15298, 7308, 5017]
    [5837, 405, 1352, 1357, 15884, 8723, 5837, 15884, 11211, 6354, 5202, 3862, 256, 11211, 6354, 6045, 15884, 3642, 2633, 9651, 15805, 14912, 6950, 6809, 11098, 3792, 112, 11818, 2620, 14686, 11387, 1, 5017]
    [3, 24, 7, 3, 20, 3, 13, 9, 13, 1, 3, 3, 9, 13, 1, 3, 3, 11, 1, 3, 3, 3, 29, 24]
    [24, 3, 1, 7, 3, 13, 24, 3, 3, 20, 9, 9, 24, 3, 20, 9, 3, 13, 1, 9, 13, 21, 20, 14, 3, 12, 27, 3, 3, 13, 3, 29, 24]
    

# <div dir='rtl'> **معماری شبکه** </div>

#### <div dir='rtl'>**مواردی که باید در نظر داشته باشیم:**  </div>

<div dir='rtl'><font size=3.5> *    ما به یک لایه نیاز داریم که یک الگوی بردار کلمه را برای کلمات ما محاسبه کند.  </div>
<div dir='rtl'> *    ما به یک لایه LSTM Bidirectional نیاز خواهیم داشت. اصلاح کننده دو طرفه نه تنها مقادیر بعدی را به LSTM مقادیر بعدی را هم بررسی می کند. </div>
<div dir='rtl'> *    ما باید پارامتر Return_sequences = True تنظیم کنیم که LSTM یک توالی را به عنوان خروجی بدهد، نه تنها مقدار نهایی. </div>
<div dir='rtl'> *    پس از لایه LSTM به یک لایه Dense (یا یک لایه کاملاً متصل) نیاز داریم که برچسب POS مناسب را مشخص کند. </div>




```python
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

model_2 = Sequential()
model_2.add(InputLayer(input_shape=(MAX_LENGTH_2, )))
model_2.add(Embedding(len(word2index_2), 128))
model_2.add(Bidirectional(LSTM(256, return_sequences=True)))
model_2.add(TimeDistributed(Dense(len(tag2index_2))))
model_2.add(Activation('softmax'))
 
model_2.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

model_2.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 327, 128)          2036352   
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 327, 512)          788480    
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 327, 30)           15390     
    _________________________________________________________________
    activation_1 (Activation)    (None, 327, 30)           0         
    =================================================================
    Total params: 2,840,222
    Trainable params: 2,840,222
    Non-trainable params: 0
    _________________________________________________________________
    

<div dir='rtl'><font size=4> قبل از شروع آموزش یک کار دیگر وجود دارد. ما باید دنباله های برچسب ها را به توالی های برچسب های رمزگذاری شده (One-Hot Encoded tags) تبدیل کنیم. این چیزی است که Dense Layer نتیجه می دهد. تابع زیر برای این کار به وجود آمده است. </div>



```python
def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)
```


```python

```

<div dir='rtl'><font size=4> یک نمونه از برچسب گذاری  one hot  در زیر نشان داده شده است. </div>


```python
cat_train_tags_y_2 = to_categorical(train_tags_y_2, len(tag2index_2))
print(cat_train_tags_y_2[0])
```

    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [1. 0. 0. ... 0. 0. 0.]
     [1. 0. 0. ... 0. 0. 0.]
     [1. 0. 0. ... 0. 0. 0.]]
    

# <div dir='rtl'> **آموزش مدل**:</div>
<div dir='rtl'><font size=4> 25 بار شبکه را آموزش می دهیم.  </div>
<div dir='rtl'> 20 درصد از اطلاعات train  را به عنوان داده های validation و ارزیابی در نظر می گیرم. </div>


```python
model_2.fit(train_sentences_X_2, to_categorical(train_tags_y_2, len(tag2index_2)), batch_size=128, epochs=25, validation_split=0.2)
```

    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/indexed_slices.py:434: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
      "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
    

    Train on 4288 samples, validate on 1073 samples
    Epoch 1/25
    4288/4288 [==============================] - 60s 14ms/step - loss: 0.7034 - accuracy: 0.8916 - val_loss: 0.2461 - val_accuracy: 0.9377
    Epoch 2/25
    4288/4288 [==============================] - 58s 13ms/step - loss: 0.2381 - accuracy: 0.9364 - val_loss: 0.2256 - val_accuracy: 0.9376
    Epoch 3/25
    4288/4288 [==============================] - 59s 14ms/step - loss: 0.2236 - accuracy: 0.9370 - val_loss: 0.2156 - val_accuracy: 0.9381
    Epoch 4/25
    4288/4288 [==============================] - 57s 13ms/step - loss: 0.2156 - accuracy: 0.9374 - val_loss: 0.2087 - val_accuracy: 0.9391
    Epoch 5/25
    4288/4288 [==============================] - 57s 13ms/step - loss: 0.2058 - accuracy: 0.9389 - val_loss: 0.1961 - val_accuracy: 0.9407
    Epoch 6/25
    4288/4288 [==============================] - 57s 13ms/step - loss: 0.1858 - accuracy: 0.9458 - val_loss: 0.1673 - val_accuracy: 0.9565
    Epoch 7/25
    4288/4288 [==============================] - 57s 13ms/step - loss: 0.1477 - accuracy: 0.9622 - val_loss: 0.1250 - val_accuracy: 0.9650
    Epoch 8/25
    4288/4288 [==============================] - 58s 14ms/step - loss: 0.1072 - accuracy: 0.9676 - val_loss: 0.0912 - val_accuracy: 0.9731
    Epoch 9/25
    4288/4288 [==============================] - 57s 13ms/step - loss: 0.0751 - accuracy: 0.9795 - val_loss: 0.0655 - val_accuracy: 0.9826
    Epoch 10/25
    4288/4288 [==============================] - 57s 13ms/step - loss: 0.0517 - accuracy: 0.9889 - val_loss: 0.0482 - val_accuracy: 0.9900
    Epoch 11/25
    4288/4288 [==============================] - 58s 13ms/step - loss: 0.0362 - accuracy: 0.9931 - val_loss: 0.0373 - val_accuracy: 0.9918
    Epoch 12/25
    4288/4288 [==============================] - 57s 13ms/step - loss: 0.0265 - accuracy: 0.9948 - val_loss: 0.0310 - val_accuracy: 0.9928
    Epoch 13/25
    4288/4288 [==============================] - 56s 13ms/step - loss: 0.0205 - accuracy: 0.9960 - val_loss: 0.0269 - val_accuracy: 0.9936
    Epoch 14/25
    4288/4288 [==============================] - 58s 13ms/step - loss: 0.0165 - accuracy: 0.9966 - val_loss: 0.0241 - val_accuracy: 0.9941
    Epoch 15/25
    4288/4288 [==============================] - 58s 13ms/step - loss: 0.0136 - accuracy: 0.9971 - val_loss: 0.0223 - val_accuracy: 0.9944
    Epoch 16/25
    4288/4288 [==============================] - 58s 14ms/step - loss: 0.0116 - accuracy: 0.9975 - val_loss: 0.0210 - val_accuracy: 0.9946
    Epoch 17/25
    4288/4288 [==============================] - 59s 14ms/step - loss: 0.0100 - accuracy: 0.9978 - val_loss: 0.0201 - val_accuracy: 0.9949
    Epoch 18/25
    4288/4288 [==============================] - 60s 14ms/step - loss: 0.0089 - accuracy: 0.9981 - val_loss: 0.0193 - val_accuracy: 0.9951
    Epoch 19/25
    4288/4288 [==============================] - 61s 14ms/step - loss: 0.0079 - accuracy: 0.9983 - val_loss: 0.0190 - val_accuracy: 0.9952
    Epoch 20/25
    4288/4288 [==============================] - 59s 14ms/step - loss: 0.0071 - accuracy: 0.9984 - val_loss: 0.0186 - val_accuracy: 0.9953
    Epoch 21/25
    4288/4288 [==============================] - 59s 14ms/step - loss: 0.0064 - accuracy: 0.9986 - val_loss: 0.0182 - val_accuracy: 0.9953
    Epoch 22/25
    4288/4288 [==============================] - 59s 14ms/step - loss: 0.0059 - accuracy: 0.9987 - val_loss: 0.0181 - val_accuracy: 0.9954
    Epoch 23/25
    4288/4288 [==============================] - 57s 13ms/step - loss: 0.0054 - accuracy: 0.9988 - val_loss: 0.0193 - val_accuracy: 0.9955
    Epoch 24/25
    4288/4288 [==============================] - 56s 13ms/step - loss: 0.0049 - accuracy: 0.9989 - val_loss: 0.0187 - val_accuracy: 0.9955
    Epoch 25/25
    4288/4288 [==============================] - 59s 14ms/step - loss: 0.0045 - accuracy: 0.9990 - val_loss: 0.0181 - val_accuracy: 0.9955
    




    <keras.callbacks.callbacks.History at 0x7f00af082ac8>



**<div dir='rtl'><font size=4> مدل خود را با داده هایی که برای آزمایش نگه داشته ایم ارزیابی کنیم:</div>**

<div dir='rtl'>  دقتی که برای این مدل از شبکه بدست آمده است: 99 درصد می باشد که دقت بسیار خوبی است  </div>


```python
scores_2 = model_2.evaluate(test_sentences_X_2, to_categorical(test_tags_y_2, len(tag2index_2)))
print(f"{model_2.metrics_names[1]}: {scores_2[1] * 100}")   # acc: 99.09751977804825
```

    3574/3574 [==============================] - 15s 4ms/step
    accuracy: 99.49482083320618
    

# <div dir='rtl'> تست مدل آموزشی
<div dir='rtl'><font size=4> پیش بینی های خود را روی داده های Test  انجام می دهیم.  </div>


```python
predictions_2 = model_2.predict(test_sentences_X_2)
print(predictions_2, predictions_2.shape)
```

    [[[3.06189235e-04 3.18723323e-06 2.61547859e-03 ... 5.76290813e-06
       2.83228583e-05 7.29372958e-04]
      [7.07319588e-04 7.58851456e-05 8.04713753e-04 ... 3.18143407e-07
       2.46761687e-04 2.40738201e-03]
      [4.21238383e-06 9.90459085e-01 1.24503160e-03 ... 3.45049648e-06
       1.05389317e-05 3.33834649e-09]
      ...
      [9.99984264e-01 1.06684167e-11 3.84171672e-09 ... 1.28131372e-08
       8.06464158e-08 8.57780719e-07]
      [9.99980569e-01 1.97545227e-11 6.02505068e-09 ... 2.30581740e-08
       1.20879918e-07 8.13786585e-07]
      [9.99975920e-01 3.36603176e-11 9.21210130e-09 ... 3.74737255e-08
       1.74363848e-07 8.17130683e-07]]
    
     [[5.81755558e-06 5.25043905e-03 7.60054565e-04 ... 4.08652522e-05
       3.29130271e-05 1.29988944e-08]
      [7.46338287e-07 7.78248761e-07 4.96931079e-07 ... 2.10512260e-10
       4.45479873e-06 2.18951300e-05]
      [1.17382072e-07 9.96011853e-01 9.16954887e-05 ... 1.00387604e-06
       4.20934794e-06 4.28877524e-11]
      ...
      [9.99984264e-01 1.06682944e-11 3.84175358e-09 ... 1.28132589e-08
       8.06459539e-08 8.57770090e-07]
      [9.99980569e-01 1.97543336e-11 6.02510797e-09 ... 2.30583499e-08
       1.20879349e-07 8.13776467e-07]
      [9.99975920e-01 3.36599290e-11 9.21217147e-09 ... 3.74740807e-08
       1.74362697e-07 8.17119030e-07]]
    
     [[3.67312285e-04 3.83509217e-07 1.04200270e-03 ... 2.12706323e-06
       2.91471188e-05 5.66143868e-03]
      [1.66685311e-06 1.04466708e-04 1.81309078e-05 ... 2.39510291e-06
       7.43702230e-06 2.05523047e-08]
      [1.66001417e-08 9.98561084e-01 7.63521020e-05 ... 8.72943531e-07
       3.71677032e-07 5.96386113e-13]
      ...
      [9.99984264e-01 1.06681322e-11 3.84157772e-09 ... 1.28126976e-08
       8.06454992e-08 8.57800330e-07]
      [9.99980569e-01 1.97540317e-11 6.02483219e-09 ... 2.30573374e-08
       1.20878880e-07 8.13805229e-07]
      [9.99975920e-01 3.36595474e-11 9.21178600e-09 ... 3.74723648e-08
       1.74361858e-07 8.17147850e-07]]
    
     ...
    
     [[9.94669608e-05 7.36836228e-05 2.26508000e-05 ... 1.26781963e-08
       4.27708619e-05 1.41694196e-04]
      [3.66056884e-05 3.56856333e-07 2.21259666e-06 ... 1.93619232e-09
       2.51648180e-05 1.36523659e-03]
      [8.64036610e-06 9.47613387e-07 4.35649696e-07 ... 2.65328370e-10
       1.45106433e-05 4.88538470e-04]
      ...
      [9.99984264e-01 1.06683360e-11 3.84176779e-09 ... 1.28133335e-08
       8.06461102e-08 8.57764348e-07]
      [9.99980569e-01 1.97544099e-11 6.02513106e-09 ... 2.30585275e-08
       1.20879577e-07 8.13771805e-07]
      [9.99975920e-01 3.36601892e-11 9.21224252e-09 ... 3.74742974e-08
       1.74363180e-07 8.17115904e-07]]
    
     [[2.59605731e-04 6.47985757e-07 2.15170112e-05 ... 3.48592835e-08
       1.12994130e-04 1.83396917e-02]
      [8.11919563e-07 5.18548295e-05 2.00558952e-06 ... 4.26967460e-07
       4.23754045e-06 1.25737927e-08]
      [3.08494151e-07 7.18700051e-07 8.29817068e-08 ... 2.38742272e-11
       3.24771145e-06 1.10682749e-05]
      ...
      [9.99984264e-01 1.06682944e-11 3.84161414e-09 ... 1.28128441e-08
       8.06461102e-08 8.57796238e-07]
      [9.99980569e-01 1.97542573e-11 6.02487793e-09 ... 2.30576020e-08
       1.20879463e-07 8.13801307e-07]
      [9.99975920e-01 3.36599290e-11 9.21185528e-09 ... 3.74727946e-08
       1.74362853e-07 8.17143928e-07]]
    
     [[4.33606783e-06 9.87810254e-01 1.34116656e-03 ... 5.95119218e-06
       8.05202399e-06 2.54715560e-09]
      [1.06636185e-06 1.07132955e-05 3.89016532e-06 ... 9.33231492e-10
       7.66545600e-06 1.88506801e-05]
      [1.40204035e-08 9.99056399e-01 1.05227024e-04 ... 2.37746747e-07
       6.02441617e-07 7.21569809e-12]
      ...
      [9.99984264e-01 1.06684375e-11 3.84175358e-09 ... 1.28132589e-08
       8.06462666e-08 8.57774182e-07]
      [9.99980569e-01 1.97545591e-11 6.02509642e-09 ... 2.30583055e-08
       1.20879818e-07 8.13779593e-07]
      [9.99975920e-01 3.36603176e-11 9.21217147e-09 ... 3.74739386e-08
       1.74363365e-07 8.17122896e-07]]] (3574, 327, 30)
    

<div dir='rtl'><font size=3> باید عملیات "برعکس" را برای to_categorical انجام دهیم، تا بتوانیم pos  ها را برای کلمات مشاهده کنیم، برای این منظور از تابع زیر استفاده می کنیم که عمل معکوس tag_to_categorical را انجام می دهد و اعداد را به فرم برچسب اولیه آنها بر میگردانیم. </div>


```python
def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences
```

<div dir='rtl'><font size=3> در اینجا چگونگی ظاهر پیش بینی ها آمده است: </div>


```python
i =logits_to_tokens(predictions_2, {i: t for t, i in tag2index_2.items()})
res = dict(zip(test_sentences_2[1], test_tags_2[1])) 
print(res)
# print(i[1][0:len(test_sentences_2[1])])
print(dict(zip(test_sentences_2[1],i[1][0:len(test_sentences_2[1])])))
```

    {'این': 'DET', 'خبر': 'N_SING', 'در': 'P', 'نشست': 'N_SING', 'شورای': 'N_SING', 'بهداشت': 'N_SING', 'استان': 'N_SING', 'تهران': 'N_SING', 'که': 'CON', 'روز': 'N_SING', 'شنبه': 'N_SING', 'محل': 'N_SING', 'استانداری': 'N_SING', 'تشکیل': 'N_SING', 'شد': 'V_PA', '،': 'DELM', 'اعلام': 'N_SING', 'گردید': 'V_PA', '.': 'DELM'}
    {'این': 'DET', 'خبر': 'N_SING', 'در': 'P', 'نشست': 'N_SING', 'شورای': 'N_SING', 'بهداشت': 'N_SING', 'استان': 'N_SING', 'تهران': 'N_SING', 'که': 'CON', 'روز': 'N_SING', 'شنبه': 'N_SING', 'محل': 'N_SING', 'استانداری': 'N_SING', 'تشکیل': 'N_SING', 'شد': 'V_PA', '،': 'DELM', 'اعلام': 'N_SING', 'گردید': 'V_PA', '.': 'DELM'}
    
