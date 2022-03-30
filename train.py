from data_processing import load_data_from_pickle, one_hot_encoding, process_massey_gesture_dataset
import torchvision.models as models
import torch
# from torchsummary import summary 


def set_requires_grad_false(model):
    for param in model.parameters():
        param.requires_grad = False

def train(model, X, y):
    # Variables
    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 256
    BATCHES = int(len(X) / BATCH_SIZE) 

    # Set optimizer
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.SGD(params_to_update, lr=LEARNING_RATE)

    # Loss Function
    loss = torch.nn.CrossEntropyLoss()

    # Train Loop
    # Variables to keep track of loss and accuracy
    train_losses = []
    # train_accuracies = []
    # valid_losses = []
    # valid_accuracies = []

    # Training loop
    print("Starting training loop...")
    for epoch in range(EPOCHS):
        avg_train_loss = 0
        for batch in range(BATCHES):
            X_batch = X[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE,] 
            y_batch = y[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE,]
            
            # Make prediction
            train_pred = model(X_batch)
            train_loss = loss(torch.squeeze(train_pred), y_batch)
            avg_train_loss += train_loss * BATCH_SIZE

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        train_losses.append(avg_train_loss/X_train.shape[0])
        # train_losses.append(100.0*train_loss.detach().numpy())
        print('Epoch %04d  Training Loss %.4f' % (epoch + 1, train_loss))   
    
    print('Saving model as "model.pt"...')
    torch.save(model, "model.pt")

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

    # Convert to floating point numbers
    y_train = y_train.to(dtype=torch.float32)

    # Create GoogLeNet model
    print("Loading GoogLeNet model...")
    model = models.googlenet(pretrained=True)
    print("Model loaded!")
    
    # Change last output layer to match output dimension
    num_classes = 24
    set_requires_grad_false(model)
    model.fc = torch.nn.Linear(1024, num_classes)

    train(model, X_train, y_train)

