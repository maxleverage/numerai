#!/usr/bin/env python

from keras import initializations
from keras.engine import Layer
from keras import backend as K 
import tensorflow as tf
import numpy as np 

class Additive(Layer):
    '''Special additive activation function
    Additive activation function combining arctan and sine using Tensorflow backend
        '''
    def __init__(self, alpha=1.0, beta=1.0, **kwargs):
        self.supports_masking = True
        self.alpha = alpha
        self.beta = beta
        super(Additive, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return self.alpha * tf.atan(x) + self.beta * tf.sin(x)

    def get_config(self):
        config = {'alpha': float(self.alpha), 'beta': float(self.beta)}
        base_config = super(Additive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Atan(Layer):
    ''' Scaled Arctan activation function using Tensorflow backend'''
    def __init__(self, alpha=1.0, **kwargs):
        self.supports_masking = True
        self.alpha = alpha
        super(Atan, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return tf.atan(x)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(Atan, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Gaussian(Layer):
    ''' Gaussian activation centered on 0, sigma to control width'''
    def __init__(self, sigma=1.0, center=0.0, **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.center = center
        super(Gaussian, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return tf.exp(-self.sigma * (x - self.center) ** 2)

    def get_config(self):
        config = {'sigma': float(self.sigma), 'center': float(self.center)}
        base_config = super(Gaussian, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SinC(Layer):
    '''  SinC activation function centered on 0 '''
    def __init__(self, alpha=1.0, **kwargs):
        self.supports_masking = True
        self.alpha = alpha
        super(SinC, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if x == 1.0:
            return 1.0
        else: return tf.sin(x) / x

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(SinC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


