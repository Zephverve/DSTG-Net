import torch 

def dct (x ):
    """
    离散余弦变换(DCT)实现
    Args:
        x: 输入张量
    """
    x_shape =x .shape 
    N =x_shape [-1 ]

    v =torch .cat ([x [...,::2 ],x [...,1 ::2 ].flip (-1 )],dim =-1 )
    Vc =torch .fft .fft (v ,dim =-1 )

    k =torch .arange (N ,device =x .device )[None ,:]*torch .pi /(2 *N )
    W_r =torch .cos (k )
    W_i =torch .sin (k )

    return Vc .real *W_r -Vc .imag *W_i 

def idct (X ):
    """
    逆离散余弦变换(IDCT)实现
    Args:
        X: 输入张量
    """
    x_shape =X .shape 
    N =x_shape [-1 ]

    k =torch .arange (N ,device =X .device )[None ,:]*torch .pi /(2 *N )
    W_r =torch .cos (k )
    W_i =torch .sin (k )

    V_t_r =X 
    V_t_i =torch .cat ([X [...,:1 ]*0 ,-X [...,1 :].flip (-1 )],dim =-1 )

    V_r =V_t_r *W_r -V_t_i *W_i 
    V_i =V_t_r *W_i +V_t_i *W_r 

    v =torch .complex (V_r ,V_i )
    x =torch .fft .ifft (v ,dim =-1 ).real 

    return torch .cat ([x [...,::2 ],x [...,1 ::2 ].flip (-1 )],dim =-1 )

__all__ =['dct','idct']