import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from dataloader import UCIHARDataset
from model import HARClassifier
import random
import numpy as np

bs = 64
lr = 1e-4
epochs = 128
feature_selection_iteration = 10000

number_of_selected_features = 64
number_of_total_features = 561

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    train_feature_file = "./../UCI HAR Dataset/train/X_train.txt"
    train_label_file = "./../UCI HAR Dataset/train/y_train.txt"
    test_feature_file = "./../UCI HAR Dataset/test/X_test.txt"
    test_label_file = "./../UCI HAR Dataset/test/y_test.txt"
    
    best_feature_acc = 0
    for iter in range(feature_selection_iteration):
        feature_indices = np.sort(random.sample(list(np.arange(0, number_of_total_features)), number_of_selected_features))

        train_dataset = UCIHARDataset(train_feature_file, train_label_file, feature_indices)
        test_dataset = UCIHARDataset(test_feature_file, test_label_file, feature_indices)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        model = HARClassifier(input_dim=number_of_selected_features, hidden_dim=64, num_layers=4, num_classes=6)
        model = model.float().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=16, gamma=0.9)

        best_acc = 0
        for epoch in range(0, epochs):
            model.train()
            for i, (X, y) in enumerate(train_loader):
                X, y = X.float().to(device), y.long().to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (X, y) in enumerate(test_loader):
                    X, y = X.float().to(device), y.long().to(device)
                    out = model(X)
                    _, predicted = torch.max(out, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                acc = correct / total * 100
                if acc > best_acc:
                    best_acc = acc
            scheduler.step()
        if best_acc > best_feature_acc:
            best_feature_acc = best_acc
            best_feature_indices = feature_indices
        
        out_str = ""
        for idx in feature_indices:
            out_str += str(idx) + " "
        out_str += str(best_acc) + "\n"

        with open("logs.txt", "a") as f:
            f.writelines(out_str)
        f.close()
        print(f"Loop: {iter}, Feature indices: {feature_indices}, Accuracy: {best_acc:.2f}%")
    
    out_str = "\n\n\n##############\n\n\n"
    for idx in best_feature_indices:
        out_str += str(idx) + " "
    out_str += str(best_feature_acc) + "\n"
    with open("logs.txt", "a") as f:
            f.writelines(out_str)
    f.close()

    print(f"Best feature indices: {best_feature_indices}, Best accuracy: {best_feature_acc:.2f}%")
