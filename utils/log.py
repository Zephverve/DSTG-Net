

from __future__ import absolute_import 

import json 
import os 
import torch 
import pandas as pd 
import numpy as np 


def save_npy_log (opt ,path ,value ,file_name ='walking'):
    path_npy =os .path .join ("point_npy",path )
    if not os .path .exists (path_npy ):
        os .makedirs (path_npy )

    file_path =os .path .join (path_npy ,f'{file_name}.npy')


    if os .path .exists (file_path ):
        existing_data =np .load (file_path )


        if existing_data .shape [1 :]==value .shape [1 :]:

            value =np .concatenate ([existing_data ,value ],axis =0 )
        else :
            print (f"警告：现有数据和新数据的形状不匹配，无法叠加")
    else :

        if len (value .shape )<2 :
            value =np .expand_dims (value ,axis =0 )


    np .save (file_path ,value )
    print (f"数据已保存到: {file_path}")


def save_csv_log (opt ,head ,value ,is_create =False ,file_name ='test'):
    if len (value .shape )<2 :
        value =np .expand_dims (value ,axis =0 )
    df =pd .DataFrame (value )
    file_path =opt .ckpt +'/{}.csv'.format (file_name )
    print (file_path )
    if not os .path .exists (file_path )or is_create :
        df .to_csv (file_path ,header =head ,index =False )
    else :
        with open (file_path ,'a')as f :
            df .to_csv (f ,header =False ,index =False )

def save_csv_eval_log (opt ,head ,value ,is_create =False ,file_name ='test'):
    if len (value .shape )<2 :
        value =np .expand_dims (value ,axis =0 )
    df =pd .DataFrame (value )
    test_sample_num =opt .test_sample_num 
    if test_sample_num ==-1 :
        test_sample_num ='all'
    file_path =opt .ckpt +'/{}_{}_eval.csv'.format (file_name ,test_sample_num )
    print (file_path )
    if not os .path .exists (file_path )or is_create :
        df .to_csv (file_path ,header =head ,index =False )
    else :
        with open (file_path ,'a')as f :
            df .to_csv (f ,header =False ,index =False )

def save_ckpt (state ,is_best =True ,file_name =['ckpt_best.pth.tar','ckpt_last.pth.tar'],opt =None ):
    file_path =os .path .join (opt .ckpt ,file_name [1 ])
    torch .save (state ,file_path )
    if is_best :
        file_path =os .path .join (opt .ckpt ,file_name [0 ])
        torch .save (state ,file_path )


def save_options (opt ):
    with open (opt .ckpt +'/option.json','w')as f :
        f .write (json .dumps (vars (opt ),sort_keys =False ,indent =4 ))
