#!/usr/bin/env python
# coding: utf-8

# In[ ]:


list_weights = [0.005, 0.01, 0.1, 0.2, 0.4, 0.5, 1]
epoch1 = 20
epoch2 = 1
validation_percentage = 0.2


# In[ ]:


import pandas as pd
import random
import pickle
from fastai.vision.all import *
from fastai.vision.core import *
from fastai.callback.all import *
from fastai.metrics import *


# In[ ]:


# read dataframe of skin lesion images
with open(r"df_data_all.pkl", "rb") as input_file:
    df_data_all = pickle.load(input_file)


# In[ ]:


def get_train(validation_percentage):
    train = df_data_all[df_data_all.test == False].copy()
    train = train[["mpx","reference","filename"]].copy()
    for index, row in train.iterrows():
        if(row["mpx"] == 1):
            train.at[index,"label2"] = "m"
        else:
            train.at[index,"label2"] = "n"
    train = add_validation(train,validation_percentage)
    return train

def add_validation2(df, validation_percentage):
    df["is_valid"] = 0
    num_pictures = len(df.index);
    while(len(df[df["is_valid"]==1]) < num_pictures*validation_percentage):
        index = int(random.random()*num_pictures);
        df.iloc[index, df.columns.get_loc('is_valid')] = 1
        filename = df.iloc[index, df.columns.get_loc('is_valid')]
        if(pd.isna(df.iloc[index, df.columns.get_loc('reference')]) == False):
            reference = df.iloc[index, df.columns.get_loc('reference')]
            df.loc[df['reference'] == reference, "is_valid"] = 1
    return df
    
def add_validation(df, validation_percentage):
    df_m = df[df["label2"].str.contains("m")].copy()
    df_n = df[df["label2"].str.contains("n")].copy()
    df_m = add_validation2(df_m, validation_percentage)
    df_n = add_validation2(df_n, validation_percentage)
    return pd.concat([df_m, df_n], ignore_index=True)

def get_dataloader(validation_percentage):
    train = get_train(validation_percentage)
    db = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=ColSplitter(),
                       get_x=ColReader("filename"),
                       get_y=ColReader("label2"),
                       item_tfms=Resize(224*2))
    db = db.new(item_tfms=RandomResizedCrop(224, min_scale=0.5),batch_tfms=aug_transforms(max_rotate=360))
    dls = db.dataloaders(train)
    return dls, train

def get_learn(weight, epoch1, epoch2):
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    weights = [1, weight]
    class_weights=torch.FloatTensor(weights).cuda()
    learn.loss_func = CrossEntropyLossFlat(weight=class_weights)
    learn.fine_tune(epoch1,freeze_epochs=epoch2)
    return learn


# In[ ]:


dls, train = get_dataloader(validation_percentage)

for weight in list_weights:
    learn = get_learn(weight, epoch1, epoch2)


# In[ ]:




