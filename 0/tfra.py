import tensorflow as tf
import tensorflow_recommenders_addons as tfra

print(tf.__path__)

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(MyModel, self).__init__()
        self.embedding = tfra.de.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
            name='tfra_embedding'
        )

    def call(self, inputs):
        return self.embedding(inputs)

# 模型参数
vocab_size = 100
embedding_dim = 64

# 实例化模型
model = MyModel(vocab_size, embedding_dim)
model.compile(optimizer='adam', loss='mse')

import numpy as np

def test_embedding_gradient_update():
    # 输入数据
    input_data = np.array([1, 2, 3, 4, 5])
    target = np.random.random((5, embedding_dim)).astype(np.float32)

    # 前向传播
    with tf.GradientTape() as tape:
        predictions = model(input_data, training=True)
        loss = tf.keras.losses.MSE(target, predictions)

    # 计算梯度
    grads = tape.gradient(loss, model.trainable_weights)

    # 检查梯度是否更新
    assert all([tf.reduce_sum(g) != 0 for g in grads]), "Gradients should not be zero"

# 运行单元测试
test_embedding_gradient_update()
