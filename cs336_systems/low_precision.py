import torch
import torch.nn as nn

# s = torch.tensor(0, dtype=torch.float32)
# for i in range(1000):
#     s += torch.tensor(0.01, dtype=torch.float32)
# print(s)

# s = torch.tensor(0, dtype=torch.float16)
# for i in range(1000):
#     s += torch.tensor(0.01, dtype=torch.float16)
# print(s)

# s = torch.tensor(0, dtype=torch.float32)
# for i in range(1000):
#     s += torch.tensor(0.01,dtype=torch.float16)
# print(s)

# s = torch.tensor(0, dtype=torch.float32)
# for i in range(1000):
#     x = torch.tensor(0.01, dtype=torch.float16)
#     s += x.type(torch.float32)
# print(s)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"model parameters dtype: {self.fc1.weight.dtype}")
        x = self.fc1(x)
        print(f"fc1 output dtype: {x.dtype}")
        x = self.relu(x)
        x = self.ln(x)
        print(f"ln output dtype: {x.dtype}")
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create the model in float32 and move to device
    model = ToyModel(10, 10).to(torch.float32).to(device)

    data = torch.randn(10, 10).to(device)
    dtype = torch.float16
    
    with torch.autocast(device_type=device.type, dtype=dtype):
        output = model(data)
        print(f"model prediction dtype: {output.dtype}")
        # just fake it
        loss = output.sum()
        print(f"loss dtype: {loss.dtype}")
        loss.backward()
        print(f"model parameters gradient dtype: {model.fc1.weight.grad.dtype}")