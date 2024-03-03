import torchvision
import summary
def print_model_summary(model:torchvision.models)->None:
    summary(model=model, input_size=(32,3,224,224), col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20, row_settings=['var_names'])
