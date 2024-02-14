import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='categorical_loss train', color='blue')
plt.plot(history.history['val_loss'], label='categorical_loss test', color='orange')
plt.plot(history.history['bbox_output_mse'], label='bbox_mse train', color='aqua')
plt.plot(history.history['val_bbox_output_mse'], label='bbox_mse test', color='red')
plt.title('График ошибок на обучающем и проверочном наборах данных')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка')
plt.legend()
plt.show()