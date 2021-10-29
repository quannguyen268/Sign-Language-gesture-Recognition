from netutil.movinet_layers import *
from netutil.Movinet_Model import *
import pickle
import tensorflow as tf
import numpy as np
# K(ab) represents a 3D kernel of size (a, b, b)
K13: KernelSize = (1, 3, 3)
K15: KernelSize = (1, 5, 5)
K33: KernelSize = (3, 3, 3)
K53: KernelSize = (5, 3, 3)

# S(ab) represents a 3D stride of size (a, b, b)
S11: KernelSize = (1, 1, 1)
S12: KernelSize = (1, 2, 2)
S22: KernelSize = (2, 2, 2)
S21: KernelSize = (2, 1, 1)
BLOCK_SPECS = {
    'a0': (
        StemSpec(filters=8, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=8,
            expand_filters=(24,),
            kernel_sizes=(K15,),
            strides=(S12,)),
        HeadSpec(project_filters=480, head_filters=2048)
    )}

input_specs = tf.keras.layers.InputSpec(shape=[None, 30, 48, 96, 3])
# inp = tf.keras.Input(shape=(30, 48, 96, 3), batch_size=1)
input_specs_dict = {'image': input_specs}

backbone = Movinet(
    model_id='a0',
    causal=True,
    use_positional_encoding=True,
    conv_type='3d_2plus1d')

model = MovinetClassifier(
    backbone,
    num_classes=64,
    kernel_regularizer=None,
    activation='swish',
    dropout_rate=0.2,
    output_states=False)

model.build([1, 30, 48, 96, 3])



#%%
inputs = np.random.rand(32, 30, 48, 96, 3)
outputs = np.zeros(shape=(32, 64))
for i in range(0, 30):
    rand = np.random.randint(64, size=(1,))
    outputs[i, rand] = 1.0

#%%
num_epochs = 3

train_steps = 32/2
total_train_steps = train_steps * num_epochs
test_steps = 32/2

loss_obj = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.1)

metrics = [
    tf.keras.metrics.TopKCategoricalAccuracy(
        k=1, name='top_1', dtype=tf.float32)]

initial_learning_rate = 0.01
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps=total_train_steps,
)
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

model.compile(loss=loss_obj, optimizer=optimizer, metrics='accuracy')

callbacks = [
    tf.keras.callbacks.TensorBoard(),
]


#%%
results = model.fit(
    inputs, outputs,
    epochs=num_epochs,
)

#%%
model.summary()