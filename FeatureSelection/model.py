import torch 
import torch.nn as nn

class HARClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(HARClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.init_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.hidden_layers = []
        for i in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.init_layer(x)
        x = self.hidden_layers(x)
        out = self.classifier(x)
        return out

if __name__ == '__main__':
    model = HARClassifier(input_dim=561, hidden_dim=128, num_layers=2, num_classes=6)
    x = torch.randn(32, 561)
    out = model(x)
    print(out.shape) 
