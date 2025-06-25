import tensorflow as tf
import numpy as np
import config

class LegacyOthelloModel(tf.keras.Model):
    """Model compatible with the TF 1.x/2.2 checkpoint format"""
    
    def __init__(self):
        super(LegacyOthelloModel, self).__init__()
        self.built_custom = False
        
    def build(self, input_shape):
        if self.built_custom:
            return
            
        # Initial conv block (Variable, batch_normalization)
        self.initial_conv = tf.keras.layers.Conv2D(
            filters=64, 
            kernel_size=3, 
            padding='same',
            use_bias=False,
            name='initial_conv'
        )
        self.initial_bn = tf.keras.layers.BatchNormalization(name='initial_bn')
        
        # 9 Residual blocks (Variable_1 to Variable_18, batch_normalization_1 to batch_normalization_18)
        self.res_blocks = []
        for i in range(9):
            block = {
                'conv1': tf.keras.layers.Conv2D(
                    filters=64, 
                    kernel_size=3, 
                    padding='same',
                    use_bias=False,
                    name=f'res_{i}_conv1'
                ),
                'bn1': tf.keras.layers.BatchNormalization(name=f'res_{i}_bn1'),
                'conv2': tf.keras.layers.Conv2D(
                    filters=64, 
                    kernel_size=3, 
                    padding='same',
                    use_bias=False,
                    name=f'res_{i}_conv2'
                ),
                'bn2': tf.keras.layers.BatchNormalization(name=f'res_{i}_bn2'),
            }
            self.res_blocks.append(block)
        
        # Policy head (Variable_19, batch_normalization_19, Variable_20, Variable_21)
        self.policy_conv = tf.keras.layers.Conv2D(
            filters=2, 
            kernel_size=1, 
            padding='same',
            use_bias=False,
            name='policy_conv'
        )
        self.policy_bn = tf.keras.layers.BatchNormalization(name='policy_bn')
        self.policy_flatten = tf.keras.layers.Flatten()
        self.policy_dense = tf.keras.layers.Dense(65, name='policy_dense')  # 65 moves
        
        # Value head (Variable_22, batch_normalization_20, Variable_23, Variable_24, Variable_25, Variable_26)
        self.value_conv = tf.keras.layers.Conv2D(
            filters=1, 
            kernel_size=1, 
            padding='same',
            use_bias=False,
            name='value_conv'
        )
        self.value_bn = tf.keras.layers.BatchNormalization(name='value_bn')
        self.value_flatten = tf.keras.layers.Flatten()
        self.value_dense1 = tf.keras.layers.Dense(64, activation='relu', name='value_dense1')
        self.value_dense2 = tf.keras.layers.Dense(1, activation='tanh', name='value_dense2')
        
        super().build(input_shape)
        self.built_custom = True
    
    def call(self, inputs, training=False):
        # Initial conv block
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = tf.nn.relu(x)
        
        # Residual blocks
        for i in range(9):
            identity = x
            
            # First conv
            x = self.res_blocks[i]['conv1'](x)
            x = self.res_blocks[i]['bn1'](x, training=training)
            x = tf.nn.relu(x)
            
            # Second conv
            x = self.res_blocks[i]['conv2'](x)
            x = self.res_blocks[i]['bn2'](x, training=training)
            
            # Skip connection
            x = tf.nn.relu(x + identity)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy, training=training)
        policy = tf.nn.relu(policy)
        policy = self.policy_flatten(policy)
        policy = self.policy_dense(policy)
        
        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value, training=training)
        value = tf.nn.relu(value)
        value = self.value_flatten(value)
        value = self.value_dense1(value)
        value = self.value_dense2(value)
        
        return policy, value

class NN:
    """Wrapper class matching the net_v2.py interface"""
    
    def __init__(self):
        self.model = LegacyOthelloModel()
        
    def f_batch(self, state_batch):
        return self.model(state_batch, training=False)
    
    def train(self, state_batch, pi_batch, z_batch):
        # For compatibility - implement if needed
        pass

def restore_legacy_checkpoint(checkpoint_path, verbose=False):
    """Restore the legacy checkpoint and return a compatible NN object"""
    
    # Create model and build it
    nn = NN()
    nn.model.build((None, 8, 8, 9))  # Input shape from checkpoint: [3, 3, 9, 64]
    
    # Create a dummy forward pass to initialize all variables
    dummy_input = tf.zeros((1, 8, 8, 9))
    _ = nn.model(dummy_input)
    
    # Map checkpoint variables to model variables
    checkpoint_variables = {}
    
    # Load checkpoint
    reader = tf.train.load_checkpoint(checkpoint_path)
    
    # Variable mapping based on the checkpoint structure
    variable_mapping = {
        # Initial conv
        'Variable': nn.model.initial_conv.kernel,
        'batch_normalization/gamma': nn.model.initial_bn.gamma,
        'batch_normalization/beta': nn.model.initial_bn.beta,
        'batch_normalization/moving_mean': nn.model.initial_bn.moving_mean,
        'batch_normalization/moving_variance': nn.model.initial_bn.moving_variance,
    }
    
    # Residual blocks
    for i in range(9):
        var_idx1 = i * 2 + 1  # Variable_1, Variable_3, Variable_5, ...
        var_idx2 = i * 2 + 2  # Variable_2, Variable_4, Variable_6, ...
        bn_idx1 = i * 2 + 1   # batch_normalization_1, batch_normalization_3, ...
        bn_idx2 = i * 2 + 2   # batch_normalization_2, batch_normalization_4, ...
        
        variable_mapping.update({
            f'Variable_{var_idx1}': nn.model.res_blocks[i]['conv1'].kernel,
            f'batch_normalization_{bn_idx1}/gamma': nn.model.res_blocks[i]['bn1'].gamma,
            f'batch_normalization_{bn_idx1}/beta': nn.model.res_blocks[i]['bn1'].beta,
            f'batch_normalization_{bn_idx1}/moving_mean': nn.model.res_blocks[i]['bn1'].moving_mean,
            f'batch_normalization_{bn_idx1}/moving_variance': nn.model.res_blocks[i]['bn1'].moving_variance,
            
            f'Variable_{var_idx2}': nn.model.res_blocks[i]['conv2'].kernel,
            f'batch_normalization_{bn_idx2}/gamma': nn.model.res_blocks[i]['bn2'].gamma,
            f'batch_normalization_{bn_idx2}/beta': nn.model.res_blocks[i]['bn2'].beta,
            f'batch_normalization_{bn_idx2}/moving_mean': nn.model.res_blocks[i]['bn2'].moving_mean,
            f'batch_normalization_{bn_idx2}/moving_variance': nn.model.res_blocks[i]['bn2'].moving_variance,
        })
    
    # Policy head
    variable_mapping.update({
        'Variable_19': nn.model.policy_conv.kernel,
        'batch_normalization_19/gamma': nn.model.policy_bn.gamma,
        'batch_normalization_19/beta': nn.model.policy_bn.beta,
        'batch_normalization_19/moving_mean': nn.model.policy_bn.moving_mean,
        'batch_normalization_19/moving_variance': nn.model.policy_bn.moving_variance,
        'Variable_20': nn.model.policy_dense.kernel,
        'Variable_21': nn.model.policy_dense.bias,
    })
    
    # Value head
    variable_mapping.update({
        'Variable_22': nn.model.value_conv.kernel,
        'batch_normalization_20/gamma': nn.model.value_bn.gamma,
        'batch_normalization_20/beta': nn.model.value_bn.beta,
        'batch_normalization_20/moving_mean': nn.model.value_bn.moving_mean,
        'batch_normalization_20/moving_variance': nn.model.value_bn.moving_variance,
        'Variable_23': nn.model.value_dense1.kernel,
        'Variable_24': nn.model.value_dense1.bias,
        'Variable_25': nn.model.value_dense2.kernel,
        'Variable_26': nn.model.value_dense2.bias,
    })
    
    # Restore variables
    for checkpoint_name, model_var in variable_mapping.items():
        try:
            checkpoint_value = reader.get_tensor(checkpoint_name)
            model_var.assign(checkpoint_value)
            if verbose:
                print(f"Restored {checkpoint_name} -> {model_var.name}")
        except Exception as e:
            print(f"Failed to restore {checkpoint_name}: {e}")
    
    if verbose:
        print("Model restoration complete!")
    return nn


if __name__ == "__main__":
    # Example usage
    nn = restore_legacy_checkpoint('./checkpoint/')
    dummy_input = tf.zeros((1, 8, 8, 9))  # Example input shape
    policy, value = nn.f_batch(dummy_input)
    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)