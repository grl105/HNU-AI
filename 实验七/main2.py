import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor

# Set MindSpore context
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


class VoiceNN(nn.Cell):
    """Neural Network for voice gender classification"""

    def __init__(self, input_size):
        super(VoiceNN, self).__init__()
        self.layer1 = nn.Dense(input_size, 64)
        self.layer2 = nn.Dense(64, 32)
        self.layer3 = nn.Dense(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.sigmoid(self.layer3(x))
        return x


class NetWithLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLoss, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label)


class VoiceGenderClassifier:
    def __init__(self):
        # Load and preprocess data
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')

        # Separate features and labels
        self.X_train = self.train_data.iloc[:, :-1]
        self.y_train = (self.train_data.iloc[:, -1] == 'male').astype(int)
        self.X_test = self.test_data.iloc[:, :-1]
        self.y_test = (self.test_data.iloc[:, -1] == 'male').astype(int)

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Convert to MindSpore tensors
        self.X_train_ms = Tensor(self.X_train_scaled.astype(np.float32))
        self.y_train_ms = Tensor(self.y_train.values.astype(np.float32))
        self.X_test_ms = Tensor(self.X_test_scaled.astype(np.float32))

    def visualize_data(self):
        """Data visualization"""
        plt.figure(figsize=(15, 5))

        # Plot feature distributions
        plt.subplot(121)
        features = self.train_data.columns[:-1]
        for i, feature in enumerate(features[:5]):  # Plot first 5 features
            male_data = self.train_data[self.train_data.iloc[:, -1] == 'male'][feature]
            female_data = self.train_data[self.train_data.iloc[:, -1] == 'female'][feature]

            plt.hist(male_data, alpha=0.5, bins=30, label=f'Male {feature}')
            plt.hist(female_data, alpha=0.5, bins=30, label=f'Female {feature}')

        plt.title('Feature Distributions by Gender')
        plt.xlabel('Feature Value')
        plt.ylabel('Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot correlation matrix
        plt.subplot(122)
        corr_matrix = self.train_data.iloc[:, :-1].corr()
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='equal')
        plt.colorbar()
        plt.title('Feature Correlation Matrix')

        plt.tight_layout()
        plt.show()

    def train_neural_network(self, epochs=1000):
        """Train neural network model"""
        print("\nTraining Neural Network...")
        self.nn_model = VoiceNN(input_size=self.X_train.shape[1])
        loss_fn = nn.BCELoss()
        net_with_loss = NetWithLoss(self.nn_model, loss_fn)
        optimizer = nn.Adam(self.nn_model.trainable_params())

        def forward_fn(data, target):
            return net_with_loss(data, target)

        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

        for epoch in range(epochs):
            loss, grads = grad_fn(self.X_train_ms, self.y_train_ms.reshape(-1, 1))
            optimizer(grads)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.asnumpy():.4f}")

        # Calculate accuracy
        predictions = (self.nn_model(self.X_test_ms).asnumpy() > 0.5).astype(int)
        self.nn_accuracy = accuracy_score(self.y_test, predictions)
        print(f"Neural Network Accuracy: {self.nn_accuracy:.4f}")

    def train_other_models(self):
        """Train and evaluate other machine learning models"""
        print("\nTraining Other Models...")

        # Logistic Regression
        self.lr_model = LogisticRegression()
        self.lr_model.fit(self.X_train_scaled, self.y_train)
        lr_pred = self.lr_model.predict(self.X_test_scaled)
        self.lr_accuracy = accuracy_score(self.y_test, lr_pred)
        print(f"Logistic Regression Accuracy: {self.lr_accuracy:.4f}")

        # Support Vector Machine
        self.svm_model = SVC()
        self.svm_model.fit(self.X_train_scaled, self.y_train)
        svm_pred = self.svm_model.predict(self.X_test_scaled)
        self.svm_accuracy = accuracy_score(self.y_test, svm_pred)
        print(f"SVM Accuracy: {self.svm_accuracy:.4f}")

        # Random Forest
        self.rf_model = RandomForestClassifier()
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        self.rf_accuracy = accuracy_score(self.y_test, rf_pred)
        print(f"Random Forest Accuracy: {self.rf_accuracy:.4f}")

    def plot_results(self):
        """Plot comparison of model performances"""
        models = ['Neural Network', 'Logistic Regression', 'SVM', 'Random Forest']
        accuracies = [self.nn_accuracy, self.lr_accuracy, self.svm_accuracy, self.rf_accuracy]

        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies)
        plt.title('Model Performance Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main():
    # Set matplotlib font settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10

    try:
        # Create classifier instance
        classifier = VoiceGenderClassifier()

        # Visualize data
        classifier.visualize_data()

        # Train and evaluate models
        classifier.train_neural_network()
        classifier.train_other_models()

        # Plot results comparison
        classifier.plot_results()

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()