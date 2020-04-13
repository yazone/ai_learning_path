## keras的K.function函数
`def function(inputs, outputs, updates=None, **kwargs)`
- 输入和输出：都是tensor或tensor的list，
- updates：是封装的更新操作opt的list;
#### 计算过程
1. K.function通过输出和输出layer，构建计算计算图
2. 调用tensorflow的tf_keras_backend.function函数，进行在tensorflow进行图计算的
3. 如何updates不为None，则计算完后按照updates的opt进行更新；


keras源码中K.function典型栗子--train_function()训练函数：
```python
self.train_function = K.function(   
                                    inputs,# 模型输出
                                    [self.total_loss] + metrics_tensors, # 输出：loss和metrics tensor
                                    updates=updates + metrics_updates,   # updates 这个暂时这么理解
                                    name='train_function',   
                                    **self._function_kwargs)
```

#### 源码
keras/backend/tensorflow_backend.py中K.function的源码：

```python
# GRAPH MANIPULATION
def function(inputs, outputs, updates=None, **kwargs):   
    if _is_tf_1():       
        v1_variable_initialization()   
        return tf_keras_backend.function(inputs, outputs, 
        updates=updates,
        **kwargs)   # 调用tensorflow的function函数
```
而tf_keras_backend.function的源码为：

```python
@keras_export('keras.backend.function')
def function(inputs, outputs, updates=None, name=None, **kwargs):
  """Instantiates a Keras function.
  Arguments:
      inputs: List of placeholder tensors.
      outputs: List of output tensors.
      updates: List of update ops.
      name: String, name of function.
      **kwargs: Passed to `tf.Session.run`.
  Returns:
      Output values as Numpy arrays.
  Raises:
      ValueError: if invalid kwargs are passed in or if in eager execution.
  """
  if ops.executing_eagerly_outside_functions():
    if kwargs:
      raise ValueError('Session keyword arguments are not support during '
                       'eager execution. You passed: %s' % (kwargs,))
    return EagerExecutionFunction(inputs, outputs, updates=updates, name=name)

  if kwargs:
    for key in kwargs:
      if (key not in tf_inspect.getfullargspec(session_module.Session.run)[0]
          and key not in ['inputs', 'outputs', 'updates', 'name']):
        msg = ('Invalid argument "%s" passed to K.function with TensorFlow '
               'backend') % key
        raise ValueError(msg)
  return GraphExecutionFunction(inputs, outputs, updates=updates, **kwargs) # 参数基原封不动传入，并进行图计算
```

#### 一个简单的栗子：
```python
import keras
from keras.models import Model
from keras.layers import Dense,Input
from keras.layers import Dropout
from keras.layers import Activation
import numpy as np
import keras.backend as K


x_val = np.random.random((1,1,20))
print(x_val)

q1 = Input(shape=(None, 20))
dense1 = Dense(64)(q1)
dense1 = Dropout(0.5)(dense1)
act1 = Activation(activation='relu')(dense1)

dense2 = Dense(64, activation='relu')(act1)
dense2 = Dropout(0.4)(dense2)
model = Model([q1], [dense2])
model.summary()

f = K.function([q1], [act1])

out = f([x_val])
print(out)
```
- input x_val:
```python
[[[0.84377892 0.1850737  0.90201927 0.6754889  0.64765142 0.87030443
   0.56530187 0.65768569 0.57400077 0.90237694 0.7225933  0.40314041
   0.84660015 0.99143577 0.98650765 0.44346917 0.25325007 0.03223878
   0.34111924 0.91580067]]]

```

- output out:
```python
[array([[[4.1230157e-01, 5.8006918e-01, 6.2487721e-01, 4.2429224e-02,
         3.0736560e-01, 3.2480699e-01, 0.0000000e+00, 6.2386423e-02,
         0.0000000e+00, 0.0000000e+00, 9.4306368e-01, 1.6242273e-01,
         0.0000000e+00, 1.1344883e-01, 0.0000000e+00, 5.4543954e-01,
         2.4292049e-01, 1.4997907e-03, 0.0000000e+00, 3.1397009e-01,
         0.0000000e+00, 0.0000000e+00, 1.5670772e-01, 9.4410157e-01,
         0.0000000e+00, 0.0000000e+00, 1.2663147e-01, 2.7181864e-02,
         3.2853875e-01, 0.0000000e+00, 1.2072456e+00, 1.6737950e-01,
         0.0000000e+00, 0.0000000e+00, 6.9894564e-01, 0.0000000e+00,
         0.0000000e+00, 7.4583179e-01, 0.0000000e+00, 8.2475281e-01,
         3.6084634e-01, 8.5202223e-01, 0.0000000e+00, 0.0000000e+00,
         5.5831301e-01, 4.4025713e-01, 1.2379751e-02, 0.0000000e+00,
         0.0000000e+00, 7.4340028e-01, 0.0000000e+00, 6.1122790e-02,
         0.0000000e+00, 3.8329601e-02, 4.3754280e-04, 8.6661118e-01,
         0.0000000e+00, 5.5022836e-02, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 6.6680539e-01, 0.0000000e+00, 0.0000000e+00]]],
      dtype=float32)]
```
