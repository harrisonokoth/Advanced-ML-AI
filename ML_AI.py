# =========================================
# Module 12: Advanced Machine Learning & AI
# =========================================

'''
This comprehensive module covers:
- Deep Learning with TensorFlow & PyTorch
- Natural Language Processing (NLP)
- Reinforcement Learning
- Computer Vision with OpenCV

Each section includes theoretical foundations, practical implementations, 
and real-world application examples.
'''


# ===========================
# 1. Deep Learning Frameworks
# ===========================

'''
Deep Learning involves neural networks with multiple layers that can learn complex patterns from data. 
Two dominant frameworks are:

- TensorFlow (Google)
- PyTorch (Facebook)
'''

# ---- TensorFlow ----
# Installation: pip install tensorflow
# Example:

import tensorflow as tf  # Import TensorFlow library for building and training models
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting graphs

def load_and_preprocess_data():
    '''Load and preprocess MNIST dataset'''
    # Load MNIST data, returns training and test sets
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1] by dividing by 255
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape images from 28x28 to 784-dimensional vectors (flatten the images)
    X_train = X_train.reshape((-1, 784))
    X_test = X_test.reshape((-1, 784))
    
    return (X_train, y_train), (X_test, y_test)  # Return preprocessed data

def build_model():
    '''Build and compile the neural network model'''
    model = tf.keras.Sequential([  # Create a Sequential model (linear stack of layers)
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),  # First hidden layer with ReLU activation and input shape 784
        tf.keras.layers.Dense(64, activation='relu'),  # Second hidden layer with ReLU activation
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer with softmax activation (for 10 classes)
    ])
    
    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
    model.compile(
        optimizer='adam',  # Optimizer to use
        loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
        metrics=['accuracy']  # Metric to track during training
    )
    
    return model  # Return the compiled model

def train_model(model, X_train, y_train, X_test, y_test):
    '''Train the model and plot results'''
    # Add early stopping to prevent overfitting by monitoring validation loss
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=3,  # Stop training if no improvement for 3 epochs
        restore_best_weights=True  # Restore weights from the best epoch
    )
    
    # Train the model on training data and validate on test data
    history = model.fit(
        X_train, y_train,  # Training data and labels
        validation_data=(X_test, y_test),  # Validation data and labels
        epochs=20,  # Number of epochs to train
        batch_size=64,  # Batch size
        callbacks=[early_stopping],  # Use early stopping
        verbose=1  # Display progress during training
    )
    
    # Plot training history (accuracy and loss)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)  # Create a subplot for accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot training accuracy
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
    plt.title('Accuracy over Epochs')  # Title for the accuracy plot
    plt.xlabel('Epoch')  # X-axis label
    plt.ylabel('Accuracy')  # Y-axis label
    plt.legend()  # Show legend
    
    plt.subplot(1, 2, 2)  # Create a subplot for loss
    plt.plot(history.history['loss'], label='Training Loss')  # Plot training loss
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot validation loss
    plt.title('Loss over Epochs')  # Title for the loss plot
    plt.xlabel('Epoch')  # X-axis label
    plt.ylabel('Loss')  # Y-axis label
    plt.legend()  # Show legend
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('training_history.png')  # Save the plot as an image file
    plt.show()  # Display the plot
    
    return model  # Return the trained model

def evaluate_model(model, X_test, y_test):
    '''Evaluate model performance'''
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  # Evaluate the model on test data
    print(f"\nTest Accuracy: {test_acc:.4f}")  # Print test accuracy
    print(f"Test Loss: {test_loss:.4f}")  # Print test loss
    
    # Show some example predictions
    predictions = model.predict(X_test[:5])  # Get predictions for the first 5 test samples
    predicted_labels = np.argmax(predictions, axis=1)  # Get the predicted class labels
    
    plt.figure(figsize=(10, 5))  # Create a new figure for predictions

    for i in range(5):  # Loop through the first 5 test images
        plt.subplot(1, 5, i+1)  # Create a subplot for each image
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')  # Display the image
        plt.title(f"Pred: {predicted_labels[i]}\nTrue: {y_test[i]}")  # Display predicted and true labels
        plt.axis('off')  # Hide axis
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('sample_predictions.png')  # Save the plot as an image file
    plt.show()  # Display the plot

def mnist_example():

    # Load and preprocess MNIST data

    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    
    # Build the neural network model

    model = build_model()
    
    # Print model summary (architecture and number of parameters)

    print("Model Summary:")
    model.summary()
    
    # Train the model on MNIST data

    print("\nTraining model on MNIST data...")
    model = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model on the test data

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":

    # Suppress TensorFlow info messages (set log level to only show errors)

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0 = all messages, 1 = no info, 2 = no warnings, 3 = no errors
    
    mnist_example()  # Run the MNIST example




'''

Discussion and interpretation of results.

1. Model Architecture:
The model consists of three layers:

First Dense Layer: 128 units with ReLU activation, resulting in 100,480 parameters.

Second Dense Layer: 64 units with ReLU activation, resulting in 8,256 parameters.

Output Layer: 10 units with softmax activation, resulting in 650 parameters.

Total parameters: 109,386, which is relatively small, allowing for a quick training process.

2. Training Process (Epoch-wise):
The training process was carried out for 20 epochs. Here are the notable results for the first few epochs:

Epoch 1:

Accuracy: 86.00%

Loss: 0.4926

Validation Accuracy: 95.80%

Validation Loss: 0.1346

At this point, the model already showed high accuracy and low loss on the validation set.

Epoch 2:

Accuracy: 96.40%

Loss: 0.1190

Validation Accuracy: 97.12%

Validation Loss: 0.0893

The model further improved its performance in both training and validation.

Epoch 3:

Accuracy: 97.69%

Loss: 0.0761

Validation Accuracy: 97.37%

Validation Loss: 0.0834

Continued improvement in accuracy, with a slight dip in validation accuracy but still very high.

Epoch 5:

Accuracy: 98.71%

Loss: 0.0426

Validation Accuracy: 97.43%

Validation Loss: 0.0799

Further improvements, with validation loss maintaining stability.

Epoch 10:

Accuracy: 99.49%

Loss: 0.0167

Validation Accuracy: 97.44%

Validation Loss: 0.0948

Very high training accuracy achieved. However, validation accuracy shows a slight plateau.

Epoch 11:

Accuracy: 99.55%

Loss: 0.0144

Validation Accuracy: 97.78%

Validation Loss: 0.0924

The model's accuracy approaches its peak, though the validation accuracy and loss remain quite stable.

3. Final Test Results:
After completing the training process, the model was evaluated on the test set:

Test Accuracy: 97.90%

Test Loss: 0.0784

These results are indicative of a well-generalized model, demonstrating good accuracy on unseen data.

Summary of Key Observations:
The model achieved high accuracy, starting at 86% and improving to nearly 99.5% by the end of training.

Training loss decreased over the epochs, which is a sign of effective learning.

The validation accuracy shows a minor fluctuation but stabilizes at around 97-98%, indicating that the model generalizes well without overfitting.

Final test accuracy of 97.90% shows that the model performs well on new, unseen data.

Validation and test losses are relatively low, indicating that the model is learning without significant overfitting or underfitting.

Statistical Significance:
The training and validation accuracy consistently increased over epochs, suggesting that the model is effectively learning the patterns from the MNIST dataset.

The gap between training accuracy and validation accuracy was minimal, implying that the model is generalizing well, as opposed to overfitting the training data.

The validation loss remained fairly stable after the early epochs, confirming that the model is not suffering from significant overfitting.

This interpretation suggests that the model is performing well, and further optimization, such as tweaking the architecture or adjusting hyperparameters, could improve performance slightly.
However, it seems to have already reached a high level of accuracy suitable for most practical purposes.

'''    

# ---- PyTorch ----
# Installation: pip install torch torchvision

def pytorch_example():
    import torch  # PyTorch base library
    import torch.nn as nn  # For neural network components
    import torch.optim as optim  # For optimization algorithms
    from torchvision import datasets, transforms  # For loading and transforming MNIST data
    from torch.utils.data import DataLoader  # For batch loading of data
    import matplotlib.pyplot as plt  # For plotting
    import numpy as np  # For numerical operations

    # Define a simple feedforward neural network
    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(784, 64)  # First hidden layer (784 input -> 64 output)
            self.fc2 = nn.Linear(64, 64)   # Second hidden layer
            self.fc3 = nn.Linear(64, 10)   # Output layer (10 classes)
            self.relu = nn.ReLU()          # ReLU activation function

        def forward(self, x):
            x = self.relu(self.fc1(x))     # Apply first layer and ReLU
            x = self.relu(self.fc2(x))     # Apply second layer and ReLU
            x = self.fc3(x)                # Output layer (logits)
            return x

    # Set device (use GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transform: convert images to tensors and flatten them
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 image to 784 vector
    ])

    # Load training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    train_losses = []
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluate model
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

    # Show some example predictions
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data = example_data.to(device)

    with torch.no_grad():
        outputs = model(example_data)
        _, preds = torch.max(outputs, 1)

    example_data = example_data.cpu().numpy()
    preds = preds.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(example_data[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {preds[i]}\nTrue: {example_targets[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_predictions_pytorch.png')
    plt.show()

if __name__ == "__main__":
        pytorch_example()


    
    '''
    Key Differences from TensorFlow:
    - More Pythonic, object-oriented approach
    - Dynamic computation graphs
    - Explicit forward pass definition
    '''

# ===================================
# 2. Natural Language Processing (NLP)
# ===================================

'''
Natural Language Processing (NLP) enables machines to understand, interpret, and generate human language in a way that is both meaningful and useful. It is a critical subfield of artificial intelligence (AI) that bridges the gap between human communication and computer understanding, allowing systems to process vast amounts of natural language data effectively.

Common applications of NLP include:

✅Text Classification: Automatically categorizing text into predefined labels or topics. This is widely used in spam detection, news categorization, and organizing customer feedback.

✅Sentiment Analysis: Determining the emotional tone behind a series of words to understand the attitudes, opinions, and emotions expressed in a text. Businesses often use this to analyze customer reviews, social media mentions, and survey responses.

✅Machine Translation: Translating text or speech from one language to another. Popular services like Google Translate use NLP to provide real-time translations that preserve the context and intent of the original message.

⏩Other noteworthy applications include chatbots and virtual assistants (like Siri and Alexa), speech recognition, text summarization, question answering, and information extraction from unstructured data.

'''

# ---- Text Preprocessing ----
# Installation: pip install nltk spacy

# === Imports ===
import ssl  # Secure Sockets Layer support for secure connections
import nltk  # Natural Language Toolkit for text processing
from nltk.tokenize import word_tokenize  # Tokenizer to split text into words
from nltk.corpus import stopwords  # Common words (e.g., "the", "is") to filter out
from nltk.stem import PorterStemmer  # Tool for reducing words to their stem
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numerical form
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier for text data
from sklearn.pipeline import make_pipeline  # Convenient way to chain preprocessing and modeling

# === SSL Context Fix (only for restricted environments) ===
try:
    _create_unverified_https_context = ssl._create_unverified_context  # Try to bypass SSL verification
    ssl._create_default_https_context = _create_unverified_https_context  # Override default context
except AttributeError:
    pass  # Do nothing if the system doesn’t support this override

# === Setup NLTK Resources ===
def setup_nltk():
    '''Ensure necessary NLTK resources are available.'''
    resources = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}  # Required resources and paths
    for res, path in resources.items():
        try:
            nltk.data.find(path)  # Check if resource exists locally
        except LookupError:
            print(f"📥 Downloading missing NLTK resource: {res}")
            nltk.download(res)  # Download the resource if not found

# === Preprocessing Example ===
def demonstrate_preprocessing():
    '''Demonstrate NLP preprocessing: tokenization, stopword removal, stemming.'''
    setup_nltk()  # Ensure required NLTK data is available

    # Sample sentence to preprocess
    text = "Natural Language Processing makes computers understand human language."

    # Tokenization: convert to lowercase and split into words
    tokens = word_tokenize(text.lower())
    print("\n🔹 Tokenization:")
    print(tokens)

    # Stopword removal: keep only meaningful words (ignore stopwords and punctuation)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    print("\n🔹 After Stopword Removal:")
    print(filtered_tokens)

    # Stemming: reduce words to their base/root form
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    print("\n🔹 After Stemming:")
    print(stemmed_tokens)

    return stemmed_tokens

# === Classification Example ===
def demonstrate_classification():
    '''Train and demonstrate a basic sentiment classifier using TF-IDF + Naive Bayes.'''

    # Sample training data with labels: 1 = Positive, 0 = Negative
    texts = [
        "I love machine learning",
        "Deep learning is fascinating",
        "I hate bugs in my code",
        "Debugging takes too much time",
        "Fixing errors is annoying",
        "I hate when my app crashes",
        "This tool makes my life easier",
        "Coding is really fun",
        "Solving problems is enjoyable",
        "Nothing works, I'm frustrated"
    ]
    labels = [1, 1, 0, 0, 0, 0, 1, 1, 1, 0]  # Corresponding sentiment labels

    # Create a pipeline that first vectorizes text using TF-IDF, then applies Naive Bayes
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(texts, labels)  # Train the model on the sample data

    # New texts to classify
    new_texts = ["I love coding", "Debugging is so hard"]
    predictions = model.predict(new_texts)  # Predict sentiment

    # Display the classification results
    print("\n🔹 Classification Results:")
    for text, label in zip(new_texts, predictions):
        sentiment = "Positive 😊" if label == 1 else "Negative 😞"
        print(f"'{text}': {sentiment}")

# === Main Program ===
if __name__ == "__main__":
    print("🚀 === NLP Demonstration Script ===")

    print("\n=== Text Preprocessing ===")
    try:
        demonstrate_preprocessing()  # Run the preprocessing demo
    except Exception as e:
        # If error occurs, guide user to install required NLTK data
        print(f"❌ Preprocessing Error: {e}")
        print("Try running manually:\n>>> import nltk\n>>> nltk.download('punkt')\n>>> nltk.download('stopwords')")

    print("\n=== Text Classification ===")
    try:
        demonstrate_classification()  # Run the classification demo
    except Exception as e:
        # If error occurs, guide user to install scikit-learn
        print(f"❌ Classification Error: {e}")
        print("Make sure scikit-learn is installed:\n>>> pip install scikit-learn")

    
    
    '''
🔰Train and predict.

In machine learning (ML), training, testing, and predicting are core steps in building models that can generalize well to new data. Let's have a look why each step is important:

1. Training
✅Purpose:
To teach the model how to recognize patterns and relationships in the data.

Why it matters:

During training, the model learns from labeled data (input-output pairs).

It adjusts internal parameters (like weights in neural networks) to minimize the difference between predicted and actual values.

The quality and quantity of training data directly influence model performance.

2. Testing
✅Purpose:
To evaluate how well the model performs on unseen data.

Why it matters:

Testing uses a separate dataset that the model hasn't seen before to assess accuracy, precision, recall, etc.

It helps detect issues like overfitting (when the model performs well on training data but poorly on new data).

Ensures the model can generalize rather than just memorize the training data.

3. Predicting
✅Purpose:
To apply the trained model to real-world or new data to make decisions or forecasts.

Why it matters:

Prediction is the final goal of most ML systems—turning input data into actionable insights.

Accurate predictions support tasks like product recommendations, fraud detection, medical diagnoses, and more.

Real-world deployment of models depends on reliable prediction performance.

    '''


    model.fit(texts, labels)
    predictions = model.predict(["I enjoy AI research"])
    print("Prediction:", predictions)

# ================================
# 3. Reinforcement Learning (RL)
# ================================

'''
RL involves training agents to make sequences of decisions by rewarding desirable behaviors. 
Key components:

- Agent: The learner/decision maker
- Environment: The world agent interacts with
- Reward: Feedback signal
'''

import gym
import numpy as np
import matplotlib.pyplot as plt

'''
In the following section, we want to create the CartPole environment with render mode specified.


CartPole is a popular reinforcement learning environment provided by OpenAI's Gym, used to demonstrate and experiment with reinforcement learning algorithms.

The Problem:
The CartPole problem involves a cart on a track that is tasked with balancing a pole on top of it. The goal is to keep the pole upright for as long as possible by applying forces to the cart.

Details of the Environment:
State Space: The environment provides a 4-dimensional state space, which includes:

Cart position: The position of the cart on the track.

Cart velocity: The speed at which the cart is moving along the track.

Pole angle: The angle of the pole with respect to vertical (i.e., how tilted the pole is).

Pole angular velocity: The rate at which the pole is rotating.

Action Space: There are two possible actions in CartPole:

0: Push the cart to the left.

1: Push the cart to the right.

Goal: The goal of the agent is to keep the pole upright for as long as possible. The agent must balance the pole by moving the cart left or right. The episode ends if:

The pole tilts past a certain angle (usually 15 degrees).

The cart moves too far off the track (beyond ±2.4 units).

A predefined number of time steps (usually 200) is reached.

Rewards: The agent receives a reward for each time step the pole remains balanced. Typically, the agent receives a reward of +1 for each time step it succeeds in keeping the pole upright.

CartPole Use in Reinforcement Learning:
The CartPole environment is often used as a benchmark problem for reinforcement learning algorithms, as it provides a relatively simple but challenging environment to test the agent's learning capability.

It's a classic control problem where the agent learns to control a system based on feedback (the state and reward).

Example: Task for the Agent
Imagine you are the agent:

You are controlling a cart on a track.

A pole is attached to the cart via a hinge.

The pole is initially upright, but you must move the cart left or right to prevent the pole from falling over.

🍖Why CartPole is Popular:
🍷Simple yet challenging: It’s simple enough for people new to reinforcement learning but also challenging enough to evaluate different algorithms.

🍷Continuous action: Even though the actions are discrete (left or right), the state space is continuous (with real-valued numbers for position, velocity, and angle), which requires the agent to learn from continuous feedback.

🍷Educational: It's used in many RL tutorials to explain key concepts like Q-learning, policy gradient methods, and value-based learning.

'''

env = gym.make('CartPole-v1', render_mode="human")
n_actions = env.action_space.n  # Number of possible actions

# Discretize the continuous state space into bins

state_bins = [
    np.linspace(-4.8, 4.8, 10),         # Cart position
    np.linspace(-5, 5, 10),             # Cart velocity
    np.linspace(-0.418, 0.418, 10),     # Pole angle
    np.linspace(-5, 5, 10)              # Pole angular velocity
]

def discretize_state(state):
    '''Convert continuous state into a tuple of discrete indices.'''

    return tuple(
        int(np.digitize(s, bins)) for s, bins in zip(state, state_bins)
    )

# Initialize the Q-table
Q = np.zeros((10, 10, 10, 10, n_actions))



'''
🛂Hyperparameters

    Hyperparameters in machine learning are configuration settings defined before training a model. Unlike parameters (like weights in neural networks), which are learned from the data, hyperparameters are set manually and control the learning process.

🔧 Examples of Hyperparameters:
✔Learning Rate – How much the model adjusts in response to the error each time it updates (too high = unstable; too low = slow learning).

✔Number of Epochs – How many times the model sees the entire training dataset.

✔Batch Size – The number of training samples used in one iteration before updating the model.

✔Number of Layers / Neurons – In neural networks, this defines the architecture.

✔Max Depth – For decision trees, it limits how deep the tree can grow.

✔Regularization Strength – Helps prevent overfitting by penalizing complexity (e.g., L1/L2 regularization).

✔K in K-Nearest Neighbors – The number of neighbors to consider when classifying a new point.

🧠 Why Hyperparameters Matter:
🔼They greatly influence the model's performance and training time.

🔼Proper tuning can help prevent overfitting or underfitting.

Selecting the right hyperparameters often involves techniques like grid search, random search, or Bayesian optimization.

⏭You can think of hyperparameters like the settings on an oven: they don’t change while baking, but choosing the right temperature and time is crucial for a good result.
'''
alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 0.1    # Exploration rate
episodes = 1000

# Store rewards for plotting
rewards = []

# Training loop
for episode in range(episodes):
    state = discretize_state(env.reset()[0])
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])        # Exploit best action

        next_obs, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_obs)

        # Q-learning update
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

print("✅ Q-learning training completed!")

# -----------------------------
# 📊 Plot total rewards over episodes
plt.plot(rewards)
plt.title("Total Rewards Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()

# ----------------------------------
# 🎥 Watch the trained agent perform
# ----------------------------------

print("🎮 Watching trained agent... (close window to exit)")
state = discretize_state(env.reset()[0])
done = False
while not done:
    env.render()
    action = np.argmax(Q[state])
    obs, _, done, _, _ = env.step(action)
    state = discretize_state(obs)
env.close()


# =============================
# 4. Computer Vision with OpenCV
# =============================

'''
Computer Vision enables machines to interpret visual data from the world.
Common tasks:
- Image classification
- Object detection
- Image segmentation
'''

# Installation: pip install opencv-python

def computer_vision_examples():
    import cv2  # OpenCV library for computer vision tasks
    import numpy as np  # NumPy for handling arrays
    import matplotlib.pyplot as plt  # Matplotlib for displaying images
    import os  # OS module to work with file paths

    # ---- Load Image ----
    image_path = 'groupofpeople.png'  # Path to the image file
    if not os.path.exists(image_path):  # Check if the image file exists
        print(f"Image file '{image_path}' not found.")  # Print error if image is missing
        return  # Exit the function

    image = cv2.imread(image_path)  # Load the image using OpenCV
    if image is None:  # Check if the image was successfully loaded
        print("Failed to load the image.")  # Print error if loading failed
        return  # Exit the function

    # ---- Convert to Grayscale ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    # ---- Edge Detection (Canny) ----
    edges = cv2.Canny(gray, 100, 200)  # Detect edges using the Canny algorithm

    # ---- Face Detection ----
    face_cascade = cv2.CascadeClassifier(  # Load the pre-trained face detector
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # XML file for face detection

    faces = face_cascade.detectMultiScale(  # Detect faces in the grayscale image
        gray, scaleFactor=1.1, minNeighbors=5)  # Parameters control detection accuracy

    print(f"Faces detected: {len(faces)}")  # Print how many faces were found

    for (x, y, w, h) in faces:  # Loop through each detected face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw a blue rectangle around each face

    # ---- Show All Results Using Matplotlib ----
    plt.figure(figsize=(15, 5))  # Create a figure with a specified size (15x5 inches)

    # Original Image with Face Detections
    plt.subplot(1, 3, 1)  # First subplot in a 1-row, 3-column grid
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB and show the image
    plt.title("Face Detection")  # Title for the first image
    plt.axis("off")  # Hide axis ticks

    # Grayscale Image
    plt.subplot(1, 3, 2)  # Second subplot
    plt.imshow(gray, cmap='gray')  # Show the grayscale version of the image
    plt.title("Grayscale")  # Title for the second image
    plt.axis("off")  # Hide axis ticks

    # Edge Detection
    plt.subplot(1, 3, 3)  # Third subplot
    plt.imshow(edges, cmap='gray')  # Show the edges detected
    plt.title("Edge Detection (Canny)")  # Title for the third image
    plt.axis("off")  # Hide axis ticks

    plt.tight_layout()  # Adjust spacing between plots
    plt.show()  # Display all three images

# Call the function to run the code above
computer_vision_examples()  # Execute the function


# ======================
# Module Summary
# ======================

'''
Key Takeaways:

1. Deep Learning:
   - TensorFlow: High-level Keras API for quick prototyping
   - PyTorch: Flexible, research-friendly framework

2. Natural Language Processing:
   - Text preprocessing (tokenization, stemming, etc.)
   - Text classification using machine learning

3. Reinforcement Learning:
   - Q-learning algorithm
   - Balance exploration vs exploitation

4. Computer Vision:
   - Image processing fundamentals
   - Object detection with Haar cascades

Practical Applications:
- Building intelligent chatbots (NLP)
- Developing self-learning game AIs (RL)
- Creating image recognition systems (CV)
- Implementing recommendation systems (DL)
'''
