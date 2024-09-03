from src.model import create_custom_model, vgg16_model
from src.data_preparation import load_train, prepare_and_load
from src.training import train_model, compile_model
from src.evaluation import evaluate_model, plot_confusion_matrix

# Load data
train_data = load_train()
val_data, val_labels = prepare_and_load(isval=True)
test_data, test_labels = prepare_and_load(isval=False)

# Create and compile model
model = create_custom_model()
model = compile_model(model)

# Train model
train_data_gen = data_gen(data=train_data, batch_size=16)
history = train_model(model, train_data_gen, val_data, val_labels)

# Evaluate model
evaluate_model(model, test_data, test_labels)
