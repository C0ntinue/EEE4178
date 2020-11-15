# Evaluation
from train import *


def test(test_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
    with torch.no_grad():
        model.eval()
        running_loss = 0
        correct_pred = 0
        n = 0

        for X, y_true in test_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass
            y_hat, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * X.size(0)
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
            test_accuracy = 100*correct_pred.float() / n
            test_loss = running_loss / len(test_loader.dataset)
    return model, test_loss, test_accuracy



if __name__ == '__main__':

    model_path = "model.pth"
    device = torch.device('cuda')

    batch_size = 128

    model= ConvNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    criterion = nn.CrossEntropyLoss()

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                         ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)



        # Evaluate
    model, test_loss, test_accuracy = test(test_loader, model, criterion, device)
    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader) * 100, test_accuracy))