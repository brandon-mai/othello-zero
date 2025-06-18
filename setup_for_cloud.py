# Complete setup for cloud environments
import tensorflow as tf
import sys
import os

def setup_for_cloud():
    """Complete setup for Colab/Kaggle environments"""
    
    # Environment detection and path setup
    if 'google.colab' in sys.modules:
        print("Setting up for Google Colab...")
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = '/content/drive/MyDrive/othello/'
    elif '/kaggle' in sys.modules or os.path.exists('/kaggle'):
        print("Setting up for Kaggle... Remember to copy checkpoint from dataset directory to working directory.")
        base_path = '/kaggle/working/'
    else:
        print("Running locally...")
        base_path = './'
    
    # Update config paths
    import config
    config.checkpoint_path = base_path + 'checkpoints/'
    config.data_path = base_path + 'data/'
    config.log_path = base_path + 'logs/'
    
    # Create directories
    for path in [config.checkpoint_path, config.data_path, config.log_path]:
        os.makedirs(path, exist_ok=True)
    
    # GPU/TPU setup
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU")
    except:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
        else:
            print("Using CPU")
    
    # Mixed precision for better performance
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    print("Setup complete!")


if __name__ == "__main__":
    setup_for_cloud()
    print("You can now run your training script.")
    # Import your model and training code here
    # from net_v2 import NetV2
    # from train import train_model
    # model = NetV2()
    # train_model(model)