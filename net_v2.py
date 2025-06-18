import numpy as np
import tensorflow as tf
import keras
from keras.layers import BatchNormalization, Conv2D, Dense, ReLU, Add, Reshape, Input
from keras.models import Model

import config


class NN:
    def __init__(self):
        self.model = OthelloModel()
    
    def f_batch(self, state_batch):
        return self.model(state_batch, training=False)
    
    def train(self, state_batch, pi_batch, z_batch):
        return self.model.train_step(state_batch, pi_batch, z_batch)


class OthelloModel(tf.keras.Model):
    def __init__(self):
        super(OthelloModel, self).__init__()
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=config.learning_rate, 
            momentum=config.momentum
        )
        
        # Initialize all layers in __init__ for trainable_variables
        self._build_layers()
        
    def _build_layers(self):
        """Initialize all layers used in the model"""
        # Initial convolutional block
        self.initial_conv = Conv2D(
            filters=config.filters_num,
            kernel_size=3,
            padding='same',
            kernel_initializer='truncated_normal',
            name='initial_conv'
        )
        self.initial_bn = BatchNormalization(name='initial_bn')
        self.initial_relu = ReLU(name='initial_relu')
        
        # Residual blocks
        self.residual_convs_1 = []
        self.residual_bns_1 = []
        self.residual_relus_1 = []
        self.residual_convs_2 = []
        self.residual_bns_2 = []
        self.residual_adds = []
        self.residual_relus_2 = []
        
        for i in range(config.residual_blocks_num):
            # First conv in residual block
            self.residual_convs_1.append(Conv2D(
                filters=config.filters_num,
                kernel_size=3,
                padding='same',
                kernel_initializer='truncated_normal',
                name=f'res_{i}_conv1'
            ))
            self.residual_bns_1.append(BatchNormalization(name=f'res_{i}_bn1'))
            self.residual_relus_1.append(ReLU(name=f'res_{i}_relu1'))
            
            # Second conv in residual block
            self.residual_convs_2.append(Conv2D(
                filters=config.filters_num,
                kernel_size=3,
                padding='same',
                kernel_initializer='truncated_normal',
                name=f'res_{i}_conv2'
            ))
            self.residual_bns_2.append(BatchNormalization(name=f'res_{i}_bn2'))
            self.residual_adds.append(Add(name=f'res_{i}_add'))
            self.residual_relus_2.append(ReLU(name=f'res_{i}_relu2'))
        
        # Policy head layers
        self.policy_conv = Conv2D(
            filters=2,
            kernel_size=1,
            padding='same',
            kernel_initializer='truncated_normal',
            name='policy_conv'
        )
        self.policy_bn = BatchNormalization(name='policy_bn')
        self.policy_relu = ReLU(name='policy_relu')
        self.policy_reshape = Reshape((-1, config.board_length * 2), name='policy_reshape')
        self.policy_dense = Dense(
            config.all_moves_num,
            kernel_initializer='truncated_normal',
            name='policy_output'
        )
        
        # Value head layers
        self.value_conv = Conv2D(
            filters=1,
            kernel_size=1,
            padding='same',
            kernel_initializer='truncated_normal',
            name='value_conv'
        )
        self.value_bn = BatchNormalization(name='value_bn')
        self.value_relu1 = ReLU(name='value_relu1')
        self.value_reshape = Reshape((-1, config.board_length), name='value_reshape')
        self.value_dense1 = Dense(
            config.filters_num,
            kernel_initializer='truncated_normal',
            name='value_dense1'
        )
        self.value_relu2 = ReLU(name='value_relu2')
        self.value_dense2 = Dense(
            1,
            kernel_initializer='truncated_normal',
            name='value_output'
        )
        
    def single_convolutional_block(self, x, training):
        """Initial convolutional block"""
        x = self.initial_conv(x)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)
        return x
    
    def residual_block(self, x, block_id, training):
        """ResNet V2 residual block using Functional API approach"""
        # Store original input for skip connection
        original_x = x
        
        # First convolution
        x = self.residual_convs_1[block_id](x)
        x = self.residual_bns_1[block_id](x, training=training)
        x = self.residual_relus_1[block_id](x)
        
        # Second convolution
        x = self.residual_convs_2[block_id](x)
        x = self.residual_bns_2[block_id](x, training=training)
        
        # Skip connection
        x = self.residual_adds[block_id]([original_x, x])
        x = self.residual_relus_2[block_id](x)
        
        return x
    
    def policy_head(self, x, training):
        """Policy head for move prediction"""
        x = self.policy_conv(x)
        x = self.policy_bn(x, training=training)
        x = self.policy_relu(x)
        x = self.policy_reshape(x)
        x = self.policy_dense(x)
        return x
    
    def value_head(self, x, training):
        """Value head for position evaluation"""
        x = self.value_conv(x)
        x = self.value_bn(x, training=training)
        x = self.value_relu1(x)
        x = self.value_reshape(x)
        x = self.value_dense1(x)
        x = self.value_relu2(x)
        x = self.value_dense2(x)
        x = tf.nn.tanh(x)
        return x
        
    def call(self, state, training=False):
        # Initial convolutional block
        x = self.single_convolutional_block(state, training)
        
        # Residual blocks
        for i in range(config.residual_blocks_num):
            x = self.residual_block(x, i, training)
            
        # Policy and value heads
        p_op = self.policy_head(x, training)
        v_op = self.value_head(x, training)
        
        return p_op, v_op
    
    @tf.function
    def train_step(self, state_batch, pi_batch, z_batch):
        with tf.GradientTape() as tape:
            p_op, v_op = self(state_batch, training=True)
            v_loss, p_loss, combined_loss = self.compute_loss(pi_batch, z_batch, p_op, v_op)
        
        gradients = tape.gradient(combined_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return p_loss, v_loss

    def compute_loss(self, pi, z, p, v):
        v_loss = tf.reduce_mean(tf.square(z - v))
        p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p, labels=pi))
        
        # L2 regularization
        l2_variables = [var for var in self.trainable_variables 
                       if 'bias' not in var.name and 'beta' not in var.name]
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in l2_variables])
        
        combined_loss = v_loss + p_loss + config.l2_weight * l2
        return v_loss, p_loss, combined_loss