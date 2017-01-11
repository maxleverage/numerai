# numerai
Basic starter code for Numerai competition

model.py - contains data preprocessing and neural network and will output predictions for class 1

get_data.py - contains a small utility function to import the training and test data, which is called from within model.py

custom_activations.py - contains 4 custom activations {Additive: linear combination of arctan(x) and sin(x), Atan: arctan(x), Gaussian: exp(-sigma * l2_norm(x, c)), SinC: sin(x) / x and 1.0 for x = 0.0}. Use by importing into model.py and adding as activation layer with NN_model.add('custom_activation').

genetic.py - implements a GA search for a subset of features using cross-validation score as objective function. Note that phenotype length defines the full feature search space we are selecting a subset from.
