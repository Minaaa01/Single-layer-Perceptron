from tkinter import *
from tkinter import ttk, Checkbutton, Entry
import Data_Preprocessing
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns

Feature_List = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "gender",
    "body_mass_g"
]
Class_list = [
    "Adelie",
    "Gentoo",
    "Chinstrap"
]
Userinfo = []


def MyFirstCombo_(event):
    Userinfo.append(MyFirstCombo.get())
    del Class_list[MyFirstCombo.current()]
    MySecondCombo = ttk.Combobox(master, state="readonly", values=Class_list)
    MySecondCombo.place(x=175, y=60)
    MySecondCombo.bind("<<ComboboxSelected>>", lambda GG: Userinfo.append(MySecondCombo.get()))


def MyThirdCombo_(event):
    del Feature_List[MyThirdCombo.current()]
    Userinfo.append(MyThirdCombo.get())
    MyFourthCombo = ttk.Combobox(master, state="readonly", values=Feature_List)
    MyFourthCombo.place(x=175, y=140)
    MyFourthCombo.bind("<<ComboboxSelected>>", lambda GG: Userinfo.append(MyFourthCombo.get()))


def Get_ETA_Value():
    Userinfo.append(ETA_Value.get())


def Get_Number_Of_Epochs_Value():
    Userinfo.append(Number_Of_Epochs_Value.get())


def Get_Bias_Value():
    Userinfo.append(Bias_Value.get())


def Decision_train_test():
    if ([Userinfo[0], Userinfo[1]] == ['Adelie', 'Gentoo']) or (
            [Userinfo[0], Userinfo[1]] == ['Gentoo', 'Adelie']):
        return Data_Preprocessing.train_c1_c2_X.replace(0, -1), Data_Preprocessing.train_c1_c2_Y.replace(0, -1), Data_Preprocessing.test_c1_c2_X.replace(0, -1), Data_Preprocessing.test_c1_c2_Y.replace(0, -1)
    elif ([Userinfo[0], Userinfo[1]] == ['Adelie', 'Chinstrap']) or (
            [Userinfo[0], Userinfo[1]] == ['Chinstrap', 'Adelie']):
        return Data_Preprocessing.train_c1_c3_X, Data_Preprocessing.train_c1_c3_Y, Data_Preprocessing.test_c1_c3_X, Data_Preprocessing.test_c1_c3_Y
    elif ([Userinfo[0], Userinfo[1]] == ['Gentoo', 'Chinstrap']) or (
            [Userinfo[0], Userinfo[1]] == ['Chinstrap', 'Gentoo']):
        return Data_Preprocessing.train_c2_c3_X.replace(0, 1), Data_Preprocessing.train_c2_c3_Y.replace(0, 1), Data_Preprocessing.test_c2_c3_X.replace(0, 1), Data_Preprocessing.test_c2_c3_Y.replace(0, 1)


def Train_Test_Model():
    if Userinfo[6] == 0:
        Bias_ = 0
    else:
        Bias_ = np.random.random_sample()

    train_x, train_y, test_x, test_y = Decision_train_test()

    Updated_Weights = Data_Preprocessing.trainPerceptron(train_x[[Userinfo[2], Userinfo[3]]],
                                                         train_y,
                                                         Userinfo[4],
                                                         Userinfo[5],
                                                         Bias_
                                                         )
    Predicted_Labels = Data_Preprocessing.testPerceptron(test_x[[Userinfo[2], Userinfo[3]]], Updated_Weights, Bias_)
    test_y = test_y.values.tolist()
    binary1 = Data_Preprocessing.confusion_matrix(Predicted_Labels, test_y)
    accuracy = Data_Preprocessing.acc(test_y, Predicted_Labels)
    plot_confusion_matrix(conf_mat=binary1)
    plt.show()
    print(f'Updated_Weights: {Updated_Weights}')
    print(f'Predicted_Labels: {Predicted_Labels}')
    print(f'Actual Labels: {test_y}')
    print('Confusion Matrix:')
    print(binary1)
    print(f'Accuracy: {accuracy} ')
    x_ = np.linspace(10, 20, 30)
    y_ = -(Updated_Weights[0] * x_)
    y_ = y_ / Updated_Weights[1]
    plt.plot(x_, y_)
    lista = train_x.columns
    f1, f2 = 0, 0
    for j in range(0, len(lista)):
        if lista[j] == Userinfo[2]:
            f1 = j
        if lista[j] == Userinfo[3]:
            f2 = j
    train_y = train_y.replace(1, Userinfo[0])
    train_y = train_y.replace(-1, Userinfo[1])
    sns.scatterplot(x=train_x.iloc[:, f1], y=train_x.iloc[:, f2], hue=train_y, palette='tab10')
    plt.show()


def Visualizing():
    Data_Preprocessing.plotting('bill_length_mm', 'bill_depth_mm')
    Data_Preprocessing.plotting('bill_length_mm', 'flipper_length_mm')
    Data_Preprocessing.plotting('bill_length_mm', 'body_mass_g')
    Data_Preprocessing.plotting('bill_depth_mm', 'flipper_length_mm')
    Data_Preprocessing.plotting('bill_depth_mm', 'body_mass_g')
    Data_Preprocessing.plotting('flipper_length_mm', 'body_mass_g')


master = Tk(className=" Single Layer Perceptron Algorithm")
master.geometry('500x500')

First_Class = StringVar()
i = Label(master, textvariable=First_Class)
First_Class.set("Select The First Class")
i.place(x=20, y=20)

MyFirstCombo = ttk.Combobox(master, state="readonly", values=Class_list)
MyFirstCombo.bind("<<ComboboxSelected>>", MyFirstCombo_)
MyFirstCombo.place(x=175, y=20)

Second_Class = StringVar()
i2 = Label(master, textvariable=Second_Class)
Second_Class.set("Select The Second Class")
i2.place(x=20, y=60)

First_Feature = StringVar()
i3 = Label(master, textvariable=First_Feature)
First_Feature.set("Select The First Feature")
i3.place(x=20, y=100)

MyThirdCombo = ttk.Combobox(master, state="readonly", values=Feature_List)
MyThirdCombo.bind("<<ComboboxSelected>>", MyThirdCombo_)
MyThirdCombo.place(x=175, y=100)

Second_Feature = StringVar()
i4 = Label(master, textvariable=Second_Feature)
Second_Feature.set("Select The Second Feature")
i4.place(x=20, y=140)

ETA = StringVar()
i5 = Label(master, textvariable=ETA)
ETA.set("Enter learning rate (eta)")
i5.place(x=20, y=180)

ETA_Value = Entry(master)
ETA_Value.place(x=175, y=180)
ETA_Value.focus_set()
ETA_Button = Button(master, text="Enter", width=8, borderwidth=4, command=Get_ETA_Value)
ETA_Button.place(x=300, y=178)

Number_Of_Epochs = StringVar()
i6 = Label(master, textvariable=Number_Of_Epochs)
Number_Of_Epochs.set("Enter number of epochs (m)")
i6.place(x=20, y=220)

Number_Of_Epochs_Value = Entry(master)
Number_Of_Epochs_Value.place(x=175, y=220)
Number_Of_Epochs_Value.focus_set()
Number_Of_Epochs_Button = Button(master, text="Enter", width=8, borderwidth=4, command=Get_Number_Of_Epochs_Value)
Number_Of_Epochs_Button.place(x=300, y=219)

Bias_Value = IntVar()
Check_Box = Checkbutton(master, text="Add bias or not", variable=Bias_Value, offvalue=0)
Check_Box.place(x=20, y=260)

Bias_Button = Button(master, text="Add", width=10, borderwidth=4, command=Get_Bias_Value)
Bias_Button.place(x=22, y=280)

Train_and_Test_Button = Button(master, text="Train and Test The Model", width=25, borderwidth=10, height=3,
                               command=Train_Test_Model)
Train_and_Test_Button.place(x=150, y=400)

Visualizing_Button = Button(master, text="Visualizing The Data Set", width=25, borderwidth=10, height=3,
                            command=Visualizing)
Visualizing_Button.place(x=150, y=320)
master.mainloop()
