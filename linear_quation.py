import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1,0,1,2,3,4],dtype=float)
ys = np.array([-3,-1,1,3,5,7],dtype=float)

# Fails without Normalization
# xs = np.array([-8756923, 0, 1, 2, 3, 48526], dtype=float)
# ys = np.array([-17513847, -1, 1, 3, 5, 97051], dtype=float)

model.fit(xs, ys, epochs=500)
print(model.predict([10]))
