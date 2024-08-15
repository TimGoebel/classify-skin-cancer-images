import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
import numpy as np

def post_processing(new_directory_path, directory):
    # Directory paths
    model_dir = os.path.join(directory, 'models')
    report_dir = os.path.join(directory, 'reports')

    # Load test data and labels
    X_test = np.load(os.path.join(new_directory_path, 'X_test.npy'))
    y_test = np.load(os.path.join(new_directory_path, 'y_test.npy'))

    # Ensure the report directory exists
    os.makedirs(report_dir, exist_ok=True)

    # List of all model files in the directory
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

    # Evaluate individual models
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        
        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Predict on the test set
        y_pred = model.predict(X_test)
        y_pred_individual = (y_pred > 0.2).astype("int32")

        # Evaluate individual model predictions
        cm_individual = confusion_matrix(y_test, y_pred_individual)
        report_individual = classification_report(y_test, y_pred_individual, target_names=['Benign', 'Malignant'], labels=[0, 1])

        # Create a figure with two subplots for the individual model
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))

        # Plot confusion matrix for the individual model
        sns.heatmap(cm_individual, annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('True')
        ax[0].set_title(f'Confusion Matrix for {model_file}')

        # Plot classification report as text for the individual model
        ax[1].text(0.01, 0.5, report_individual, fontsize=12, va='center', ha='left')
        ax[1].axis('off')
        ax[1].set_title('Classification Report')

        # Save the figure for the individual model
        output_path_individual = os.path.join(report_dir, f'{os.path.splitext(model_file)[0]}_report.png')
        plt.tight_layout()
        plt.savefig(output_path_individual)
        plt.close()

        print(f'Report saved for individual model {model_file} at {output_path_individual}')

    # Generate all possible pairs of models for ensemble
    model_pairs = list(itertools.combinations(model_files, 2))

    # Loop through each pair of models for ensemble
    for model_file1, model_file2 in model_pairs:
        # Load the models
        model_path1 = os.path.join(model_dir, model_file1)
        model_path2 = os.path.join(model_dir, model_file2)
        
        model1 = tf.keras.models.load_model(model_path1)
        model2 = tf.keras.models.load_model(model_path2)

        # Predict on the test set with both models
        y_pred1 = model1.predict(X_test)
        y_pred2 = model2.predict(X_test)

        # Average the predictions for ensemble
        y_pred_ensemble = (y_pred1 + y_pred2) / 2

        # Convert probabilities to class labels
        y_pred_final = (y_pred_ensemble > 0.2).astype("int32")

        # Evaluate the ensemble predictions
        cm_ensemble = confusion_matrix(y_test, y_pred_final)
        report_ensemble = classification_report(y_test, y_pred_final, target_names=['Benign', 'Malignant'], labels=[0, 1])

        # Create a figure with two subplots for the ensemble model
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))

        # Plot confusion matrix for the ensemble model
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('True')
        ax[0].set_title(f'Confusion Matrix for Ensemble: {model_file1} & {model_file2}')

        # Plot classification report as text for the ensemble model
        ax[1].text(0.01, 0.5, report_ensemble, fontsize=12, va='center', ha='left')
        ax[1].axis('off')
        ax[1].set_title('Classification Report')

        # Save the figure for the ensemble model
        output_filename = f'ensemble_{os.path.splitext(model_file1)[0]}_{os.path.splitext(model_file2)[0]}.png'
        output_path_ensemble = os.path.join(report_dir, output_filename)
        plt.tight_layout()
        plt.savefig(output_path_ensemble)
        plt.close()

        print(f'Ensemble report saved for models {model_file1} & {model_file2} at {output_path_ensemble}')
