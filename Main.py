from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy.linalg import norm
from numpy import dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from numpy import array
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Convolutional Neural Network Based Text Steganalysis")
main.geometry("1300x1200")

global filename
global trainX
global trainy
global testX
global model
global tokenizer
global length
global accuracy

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def process_docs(directory, is_trian):
    documents = list()
    for filename in listdir(directory):
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents


def process_text(text):
    documents = list()
    tokens = clean_doc(text)
    documents.append(tokens)
    return documents



def preprocess():
    global trainX
    global trainy
    docs1 = process_docs(filename+'/topic1', True)
    docs2 = process_docs(filename+'/topic2', True)
    trainX = docs1 + docs2
    trainy = [0 for _ in range(len(docs1))] + [1 for _ in range(len(docs2))]
    text.delete('1.0', END)
    text.insert(END,"Features from dataset\n\n");
    text.insert(END,str(trainX))
    
    
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded    

def cnn():
    global accuracy
    global trainX
    global model
    global tokenizer
    global length
    text.delete('1.0', END)
    tokenizer = create_tokenizer(trainX)
    length = max_length(trainX)
    vocab_size = len(tokenizer.word_index) + 1
    text.insert(END,'Max document length : '+str(length)+"\n")
    text.insert(END,'Vocabulary size : '+str(vocab_size)+"\n")
    trainX = encode_text(tokenizer, trainX, length)
    text.insert(END,"Features Embedding Vector\n\n")
    text.insert(END,str(trainX)+"\n\n")

    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    merged = concatenate([flat1, flat2, flat3])
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit([trainX,trainX,trainX], array(trainy), epochs=10, batch_size=1)
    text.insert(END,"CNN Steganalysis model generated. See black console for details\n")
    loss, acc = model.evaluate([trainX,trainX,trainX], array(trainy), verbose=0)
    text.insert(END,'Propose LS-CNN Accuracy : '+str(acc*100))
    accuracy = acc * 100
    
def predictSteg(vec1, vec2):
    vector1 = np.asarray(vec1)
    vector2 = np.asarray(vec2)
    return dot(vector1, vector2)/(norm(vector1)*norm(vector2))

def predict():
    global testX
    text.delete('1.0', END)
    input_sentence = simpledialog.askstring("Enter your sentence here  for steg analysis detection", "Enter your sentence here  for steg analysis detection")
    testX = process_text(input_sentence)
    testX = encode_text(tokenizer, testX, length)
    ypred = model.predict([testX,testX,testX])
    result = 0
    classname = -1
    for i in range(len(trainX)):
        score = predictSteg(trainX[i],testX[0])
        if score > result:
            result = score
            classname = i
    if trainy[classname] == 0:
        text.insert(END,input_sentence+" contains no steg text")
    elif trainy[classname] == 1:
        text.insert(END,input_sentence+" contains steg text")
    else:
        text.insert(END,"Unable to understand. Given sentence out of trained model")

def graph():
    height = [accuracy,90]
    bars = ('Propose LC-CNN Accuracy', 'Existing T-LEX Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   
                    
    
font = ('times', 16, 'bold')
title = Label(main, text='Convolutional Neural Network Based Text Steganalysis',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Text File with Sentences", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

modelButton = Button(main, text="Preprocess and Convert Text To Indexes", command=preprocess)
modelButton.place(x=50,y=200)
modelButton.config(font=font1)

svmButton = Button(main, text="Run CNN Algorithm & Embeded Words", command=cnn)
svmButton.place(x=50,y=250)
svmButton.config(font=font1)

naiveButton = Button(main, text="Predict Steg Analsysis from sentence", command=predict)
naiveButton.place(x=50,y=300)
naiveButton.config(font=font1)

naiveButton = Button(main, text="Accuracy Graph", command=graph)
naiveButton.place(x=50,y=350)
naiveButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
