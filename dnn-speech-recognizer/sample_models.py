from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    #bn_rnn = ...
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    last_input = input_data #save the last input to feed into next neural network layer
    arr_rrn = []
    arr_batch_norm = []
    for x in range(0, recur_layers):
        rnn_current = GRU(units, activation='relu', return_sequences=True,implementation=2, name="rnn_layer_{}".format(x))(last_input)
        arr_rrn.append(rnn_current)
        curr_batch_norm = BatchNormalization(name="bn_layer_{}".format(x))(rnn_current)
        arr_batch_norm.append(curr_batch_norm)
        last_input = arr_batch_norm[x]
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(last_input)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2, name = 'bidir_rnn'))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, filters, kernel_size, conv_stride, conv_border_mode, output_dim=29):
    """ Build a deep network for speech 
    """
    # Develop this: Input --> {CNN-->BatchNorm}-->{Bidir_RNN1-->Dropout-->BatchNorm_RNN1}-->{Bidir_RNN2-->Dropout-->BatchNorm_RNN2}-->{Bidir_RNN3-->Dropout-->BatchNorm_RNN3} -->TimeDistributed(Dense)-->Softmax_activation 
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # TODO: Specify the layers in your network
    #First lets add a CNN layer for the feature extraction
    cnn_1d = Conv1D(filters, kernel_size, strides = conv_stride, padding=conv_border_mode,activation='relu', name='cnn_1d')(input_data)
    #Lets add batch normalization to this Cnn
    batch_norm_cnn = BatchNormalization(name='batch_norm_cnn')(cnn_1d)
        
    #Add first bidrectional recurrent layer, with dropout and batch normalization
    rnn1 = Bidirectional(GRU(units, activation='relu', return_sequences=True, name='rnn1', implementation=2), merge_mode="concat")(batch_norm_cnn)
    dropout_rnn1 = Dropout(0.2)(rnn1)
    bn_rnn1 = BatchNormalization(name='bn_rnn1')(dropout_rnn1)
    
    #Add second bidrectional recurrent layer, with dropout and batch normalization
    
    rnn2 = Bidirectional(GRU(units, activation='relu', return_sequences=True, name='rnn2', implementation=2), merge_mode="concat")(bn_rnn1)
    dropout_rnn2 = Dropout(0.2)(rnn2)
    bn_rnn2 = BatchNormalization(name='bn_rnn2')(dropout_rnn2)
    
    #Add a third one similarly 
    rnn3 = Bidirectional(GRU(units, activation='relu', return_sequences=True, name='rnn3', implementation=2), merge_mode="concat")(bn_rnn2)
    dropout_rnn3 = Dropout(0.2)(rnn3)
    bn_rnn3 = BatchNormalization(name='bn_rnn3')(dropout_rnn3)

    #TimeDistributed dense layer
    time_dist_dense = TimeDistributed(Dense(output_dim))(bn_rnn3)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dist_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length: This would be similar to the CNN model used above. 
    
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model