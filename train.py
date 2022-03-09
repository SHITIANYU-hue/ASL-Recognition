from data_processing import load_data_from_pickle
import torchvision.models as models
import torch
# from torchsummary import summary 


def set_requires_grad_false(model):
    for param in model.parameters():
        param.requires_grad = False

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

    # Change input data to conform to CNN input shape
    X_train = torch.permute(X_train, (0, 3, 1, 2))

    # Convert to floating point numbers
    y_train = y_train.to(dtype=torch.float32)

    # Variables
    num_classes = 24
    lr = 0.001
    num_epochs = 10
    
    # Create GoogLeNet model
    print("Loading GoogLeNet model...")
    model = models.googlenet(pretrained=True)
    print(model) 
    
    # Change last output layer to match output dimension
    set_requires_grad_false(model)
    model.fc = torch.nn.Linear(1024, num_classes)

    # Set optimizer
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = torch.optim.SGD(params_to_update, lr=lr)

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
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Model predictions
        train_pred = torch.squeeze(model(X_train))

        # Get loss
        train_loss = loss(train_pred, y_train)

        train_loss.backward()
        optimizer.step()

        # Append loss
        train_losses.append(100.0*train_loss.detach().numpy())

        print('Epoch %04d  Training Loss %.4f' % (epoch + 1, train_loss))

