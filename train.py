from data_processing import load_data_from_pickle, one_hot_encoding, process_massey_gesture_dataset
import torchvision.models as models
import torch
import matplotlib.pyplot as plt

def set_requires_grad_false(model):
    for param in model.parameters():
        param.requires_grad = False

def train(model, X_train, y_train, X_valid, y_valid):
    # Variables
    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 256
    
    TRAIN_BATCHES = int(len(X_train) / BATCH_SIZE) 
    VALID_BATCHES = int(len(X_valid) / BATCH_SIZE)

    # Set optimizer
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.SGD(params_to_update, lr=LEARNING_RATE)

    # Loss Function
    loss = torch.nn.CrossEntropyLoss()

    # Variables to keep track of loss and accuracy
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    # Training loop
    print("Starting training loop...")
    for epoch in range(EPOCHS):
        avg_train_loss = 0      # (averaged) training loss per batch
        avg_valid_loss =  0     # (averaged) validation loss per batch
        train_acc = 0           # training accuracy per batch
        valid_acc = 0           # validation accuracy per batch

        for batch in range(TRAIN_BATCHES):
            X_batch = X_train[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE,] 
            y_batch = y_train[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE,]
            
            # Make prediction
            train_pred = model(X_batch)
            print("train_pred (before):", train_pred)

            train_pred = torch.nn.functional.one_hot(torch.argmax(train_pred, dim=1), num_classes=24)

            train_loss = loss(torch.squeeze(train_pred.type(torch.FloatTensor)), y_batch)
            avg_train_loss += train_loss * BATCH_SIZE
            train_acc += (train_pred == y_batch).sum()
            print("train_pred (after):", train_pred)
            print("y_batch:", y_batch)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        train_losses.append(avg_train_loss/X_train.shape[0])
        train_accuracies.append(train_acc/X_train.shape[0])

        # Repeat with validation set
        for batch in range(VALID_BATCHES):
            X_batch = X_valid[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE,] 
            y_batch = y_valid[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE,]
            
            # Make prediction
            valid_pred =  model(X_batch)
            valid_pred = torch.nn.functional.one_hot(torch.argmax(valid_pred, dim=1), num_classes=24).float()
            valid_loss = loss(torch.squeeze(valid_pred.type(torch.FloatTensor)), y_batch)
            avg_valid_loss += valid_loss * BATCH_SIZE
            valid_acc += (valid_pred == y_batch).sum()

            optimizer.zero_grad()
            valid_loss.backward()
            optimizer.step()

        valid_losses.append(avg_valid_loss/X_valid.shape[0])
        valid_accuracies.append(valid_acc/X_valid.shape[0])

        # train_losses.append(100.0*train_loss.detach().numpy())
        print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/X_train.shape[0], avg_valid_loss/X_valid.shape[0], 100*train_acc/X_train.shape[0], 100*valid_acc/X_valid.shape[0]))

        # print('Epoch %04d  Training Loss %.4f' % (epoch + 1, train_loss))   
    
    #Plot training loss
    plt.title("Train vs Validation Loss")
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Train vs Validation Accuracy")
    plt.plot(train_accuracies, label="Train")
    plt.plot(valid_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='best')
    plt.show()

    print('Saving model as "model.pt"...')
    torch.save(model, "model.pt")

class ASLNet(torch.nn.Module):
    def __init__(self):
        super(ASLNet, self).__init__()
        self.model = models.googlenet(pretrained=True)
        set_requires_grad_false(self.model)
        self.model.fc = torch.nn.Linear(1024, 24)

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    # Load Data
    print("Loading dataset...")
    data = load_data_from_pickle("data.pkl")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_valid = data["X_valid"]
    y_valid = data["y_valid"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    print("Data loaded!")

    # Change input data to conform to CNN input shape
    X_train = torch.permute(X_train, (0, 3, 1, 2))
    X_valid = torch.permute(X_valid, (0, 3, 1, 2))
    X_test = torch.permute(X_test, (0, 3, 1, 2))

    # Convert to floating point numbers
    X_train = X_train.to(dtype=torch.float32)
    X_valid = X_valid.to(dtype=torch.float32)
    X_test = X_test.to(dtype=torch.float32)

    y_train = y_train.to(dtype=torch.float32)
    y_valid = y_valid.to(dtype=torch.float32)
    y_test = y_test.to(dtype=torch.float32)

    # Create GoogLeNet model
    print("Loading GoogLeNet model...")
    # model = models.googlenet(pretrained=True)
    model = ASLNet()
    print("Model loaded!")
    
    # Change last output layer to match output dimension
    # num_classes = 24
    # model.fc = torch.nn.Linear(1024, num_classes)

    # Train the model
    X_train = X_train[0:1000]
    y_train = y_train[0:1000]
    X_valid = X_valid[0:500]
    y_valid = y_valid[0:500]
    train(model, X_train, y_train, X_valid, y_valid)

    # model = torch.load("results/model.pt")

    with torch.no_grad():
        y_predict = model(X_test)
        accuracy=((y_predict == y_test).sum()) / (y_test.shape[0])
        print(f"Test Accuracy: {accuracy.item(): .4f}")
