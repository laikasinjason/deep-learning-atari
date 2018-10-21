from keras.models import model_from_yaml
def save_model(model):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("atari_dqn_model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("atari_dqn_model.h5")
    print("Saved model to disk")