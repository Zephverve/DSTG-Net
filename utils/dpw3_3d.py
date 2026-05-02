from torch .utils .data import Dataset 
import pickle as pkl 
import numpy as np 
from os import walk 
import scipy .io as sio 
from utils import data_utils 
from matplotlib import pyplot as plt 


import torch 

class Datasets (Dataset ):

    def __init__ (self ,opt ,actions =None ,split =0 ):

        path_to_data =opt .data_dir 
        input_n =opt .input_n 
        output_n =opt .output_n 
        self .in_n =input_n 
        if output_n ==13 :
            output_n =25 
            self .out_n =25 
            self .downsample =True 




        else :
            self .out_n =output_n 
            self .downsample =False 

        if split ==1 :
            their_input_n =50 
        else :
            their_input_n =input_n 

        seq_len =their_input_n +output_n 

        if split ==0 :
            self .data_path =path_to_data +'/train/'
        elif split ==1 :
            self .data_path =path_to_data +'/validation/'
        elif split ==2 :
            self .data_path =path_to_data +'/test/'
        all_seqs =[]
        files =[]


        for (dirpath ,dirnames ,filenames )in walk (self .data_path ):
            files .extend (filenames )
        for f in files :
            with open (self .data_path +f ,'rb')as f :
                data =pkl .load (f ,encoding ='latin1')
                joint_pos =data ['jointPositions']
                for i in range (len (joint_pos )):
                    seqs =joint_pos [i ]
                    seqs =seqs -seqs [:,0 :3 ].repeat (24 ,axis =0 ).reshape (-1 ,72 )
                    n_frames =seqs .shape [0 ]
                    fs =np .arange (0 ,n_frames -seq_len +1 )
                    fs_sel =fs 
                    for j in np .arange (seq_len -1 ):
                        fs_sel =np .vstack ((fs_sel ,fs +j +1 ))
                    fs_sel =fs_sel .transpose ()
                    seq_sel =seqs [fs_sel ,:]
                    if len (all_seqs )==0 :
                        all_seqs =seq_sel 
                    else :
                        all_seqs =np .concatenate ((all_seqs ,seq_sel ),axis =0 )



        self .dim_used =np .array (range (3 ,all_seqs .shape [2 ]))

        all_seqs =all_seqs [:,(their_input_n -input_n ):,:]
        self .all_seqs =all_seqs *1000 

    def __len__ (self ):
        return np .shape (self .all_seqs )[0 ]

    def __getitem__ (self ,item ):
        fs =np .arange (0 ,self .in_n )
        fs1 =np .arange (self .in_n +1 ,self .in_n +self .out_n )[::2 ]
        fs2 =np .arange (self .in_n +self .out_n -1 ,self .in_n +self .out_n )
        if self .downsample :
            if self .out_n ==25 :
                fs1 [7 ]=24 
                fs =np .concatenate ((fs ,fs1 ,fs2 ))
                src =self .all_seqs [item ,fs ]
                return src 
            elif self .out_n ==30 :
                fs1 =np .arange (self .in_n +1 ,self .in_n +self .out_n )[::3 ]
                fs =np .concatenate ((fs ,fs1 ,fs2 ))
                src =self .all_seqs [item ,fs ]
                return src 

        else :
            fs =np .arange (0 ,0 +self .in_n +self .out_n )
            src =self .all_seqs [item ,fs ]
            return src 

