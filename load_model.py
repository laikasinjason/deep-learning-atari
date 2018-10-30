from keras.models import model_from_yaml

def load_model(atari_model):
    # load YAML and create model
    yaml_file = open('atari_dqn_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("atari_dqn_model.h5")
    atari_model.model = loaded_model
    print("Loaded model from disk")
    
    loaded_model2 = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model2.load_weights("atari_dqn_model.h5")
    atari_model.target_model = loaded_model
    
    print("Loaded target model")
    # compile the loaded model before further use