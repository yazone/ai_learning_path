# Keras单机多GPU并行训练
&emsp;&emsp;在多个GPU上运行单个模型有两种方法：**数据并行**和**设备并行**。


&emsp;&emsp;在大多数情况下，一般使用数据并行的方式，另外建议使用TensorFlow后端进行此操作。

## 数据并行（Data parallelism）
&emsp;&emsp;数据并行在每个设备上复制一次目标模型，并使用每个副本处理输入数据的不同部分。Keras有一个内置的实用程序```Keras.utils.multi-gpu-gpu-model```，它可以在任何模型进行数据并行训练，并在高达8个gpu的情况下实现准线性加速。

### multi_gpu_model函数
函数原型如下：

``` python
keras.utils.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False)
```

该函数能够在不同的GPU上复制模型，具体来说，该功能实现了单机多GPU数据并行。它的工作方式如下：

- 每个GPU单独加载模型副本；
- 将输入数据拆分为多个子批量(sub-batch)；
- 每个模型副本独立加载一份子批量数据进行训练；
- 将各个GPU上训练的结果（在CPU上）进行连接得出整体。

例如：如果您的batch size是64，并且使用```gpus=2```，那么我们将输入分成两个32个样本的子批，在一个GPU上处理每个子批，然后返回64个已处理样本的整批计算结果。

这个函数能实现高达8个gpu的准线性加速。

&emsp;&emsp;此函数暂时仅在TensorFlow后端可用。

参数：

- model：Keras模型实例。为了避免OOM错误，这个模型可以建立在CPU上，例如（参见下面的使用示例）。

- gpus:Integer>=2或整数列表，要在其上创建模型副本的GPU数量或GPU ID列表。

- cpu_merge：一个布尔值，用于标识是否强制在cpu上合并模型权重。

- cpu_relocation：一个布尔值，用于标识是否在cpu上创建模型的权重。如果模型未在定义设备作用域，仍可以通过此选项来进行更改。

返回：

- 一个Keras模型实例，它将工作分布在多个gpu上进行。

### 示例
&emsp;&emsp;下面是一个简单的例子：
``` python
from keras.utils import multi_gpu_model

# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

### 模型保存
1、使用model.save()或model.save_weights()

&emsp;&emsp;要保存训练好的模型，使用的是传入multi_gpu_model的参数model对应的模型实例，而不是multi_gpu_model的返回值模型实例。例如上面示例中使用model.save()而不是parallel_model.save()。

2、使用keras.callbacks.ModelCheckpoint

假设使用ModelCheckpoint保存代码如下：
``` python
saveBestModel = keras.callbacks.ModelCheckpoint(SAVED_MODEL_PATH, monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=True, mode='auto')

parallel_model.fit(x, y, epochs=20, batch_size=256,callbacks=[saveBestModel])
```

&emsp;&emsp;由于```parallel_model.fit(callbacks=[saveBestModel])```中的ModelCheckpoint实例默认使用的是```parallel_model```，因此需要改写ModelCheckpoint子类中的```set_model()```：

``` python
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
```

使用时：
``` python
saveBestModel = ParallelModelCheckpoint(model,SAVED_MODEL_PATH, monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=True, mode='auto')
parallel_model.fit(x, y, epochs=20, batch_size=256,callbacks=[saveBestModel])
```

## 设备并行（Device parallelism）
&emsp;&emsp;设备并行是指在不同的设备上运行同一模型的不同部分。它最适用于具有并行体系结构的模型，例如具有两个分支的模型。这可以通过使用TensorFlow设备范围来实现。下面是一个简单的例子：

```python
# Model where a shared LSTM is used to encode two different sequences in parallel
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# Process the first sequence on one GPU
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# Process the next sequence on another GPU
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# Concatenate results on CPU
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

引用：\
[1] https://keras.io/utils/#multi_gpu_model \
[2] https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus
