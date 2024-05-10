def no_tl(features, data_clean, index, lag, n_ahead, n_layer, learning_rate, epochs, batch_size):
    """
    A method to create a Multivariate LSTM standard model with no TL
    """
    data_clean_s, train_mean, train_std, test_share = scale_data(features, data_clean, index)
    X, Y = create_X_Y(data_clean_s.values, lag=lag, n_ahead=n_ahead)
    n_ft = X.shape[2]

    # Spliting into train and test sets 
    Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share))], Y[0:int(X.shape[0] * (1 - test_share))]
    Xval, Yval = X[int(X.shape[0] * (1 - test_share)):], Y[int(X.shape[0] * (1 - test_share)):]

    print(f"Shape of training data: {Xtrain.shape}")
    print(f"Shape of the target data: {Ytrain.shape}")

    print(f"Shape of validation data: {Xval.shape}")
    print(f"Shape of the validation target data: {Yval.shape}")

    model = NNMultistepModel(
        X=Xtrain,
        Y=Ytrain,
        n_outputs=n_ahead,
        n_lag=lag,
        n_ft=n_ft,
        n_layer=n_layer,
        Xval=Xval,
        Yval=Yval,
    )
    model.model.summary()

    history = model.train(lr=learning_rate,epochs=epochs, batch_size=batch_size)
    training_time = model.get_training_time()
    
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    n_epochs = range(len(loss))
    plt.figure(figsize=(9, 7))
    plt.plot(n_epochs, loss, 'r', label='Training loss', color='blue')
    if val_loss is not None:
        plt.plot(n_epochs, val_loss, 'r', label='Validation loss', color='red')
    plt.legend(loc=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.show()

    # Forecasting on all the samples in the validation set 
    forecast = model.predict(Xval)
    target_index = 0

    # Reversed Metrics
    reversed_forecast = forecast * train_std[target_index] + train_mean[target_index]
    reversed_Yval = Yval * train_std[target_index] + train_mean[target_index]

    model_name = f'{building_name}_std'

    append_model_metrics(model_name, reversed_forecast, reversed_Yval, target_index, training_time)
