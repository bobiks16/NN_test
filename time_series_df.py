def prepare_data(data, target, lookback):
    x = []
    y = []
    for i in range(len(data) - lookback):
        x.append(data[i:i+lookback])
        y.append(target[i+lookback])
    return np.array(x), np.array(y)

x, y = prepare_data(data.values, target.values, 7)