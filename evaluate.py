from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    generate_classification_report,
)
from src.model import vgg16_model
from src.data_preparation import prepare_and_load

# Load test data
test_data, test_labels = prepare_and_load(isval=False)

# Load the trained model
model = vgg16_model()
model.load_weights("best_model.h5")

# Evaluate model
evaluate_model(model, test_data, test_labels)

# Predict and generate reports
predictions = model.predict(test_data, batch_size=16)
predictions = np.argmax(predictions, axis=-1)
labels = np.argmax(test_labels, axis=-1)
plot_confusion_matrix(labels, predictions, classes=["NORMAL", "PNEUMONIA"])
generate_classification_report(labels, predictions)
