import torch
import torch.nn as nn

# pytorch encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# pytorch decoder model
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# pytorch autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.output_dim)
        self.decoder = Decoder(self.output_dim, self.hidden_dim, self.input_dim)

    

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def num_parameters(self):
        return self.encoder.num_parameters() + self.decoder.num_parameters()

# create custom pytorch dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# autoencoder data loader
def autoencoder_data_loader(data, batch_size, shuffle):
    # define data loader
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    # return data loader
    return data_loader

# autoencoder training
def train_autoencoder(model, train_loader, test_loader, num_epochs, learning_rate, device):
    losses=[]
    # define loss function
    criterion = torch.nn.MSELoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    counter=0
    # train model
    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss = 0
        for data in train_loader:
            # get data
            data = data.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            # calculate loss
            train_loss += loss.item()
        # test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                # get data
                data = data.to(device)
                # forward
                outputs = model(data)
                loss = criterion(outputs, data)
                # calculate loss
                test_loss += loss.item()
        # print loss
        if counter%100==0: print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss, test_loss))
        counter+=1
        losses.append([train_loss, test_loss])
    return losses

    



raw_data=torch.tensor([
    [1,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0],
    [1,1,1,0,0,0,0,0,0,0,0],
    [1,1,1,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,0,1,1,1,1],
    [0,0,0,0,0,0,1,1,1,1,1],
],dtype=torch.float32)

#from PIL import Image
import matplotlib.pyplot as plt
# define dataloader
train_data = CustomDataset(raw_data)
train_loader = autoencoder_data_loader(train_data, batch_size=8, shuffle=True)
test_loader = autoencoder_data_loader(train_data, batch_size=8, shuffle=False)
model=Autoencoder(11, 16, 8)
print(model.num_parameters())
traindevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(traindevice)
print('device', traindevice)
losses=train_autoencoder(model, train_loader, test_loader, num_epochs=1000, learning_rate=0.005, device=traindevice)
fig, ax = plt.subplots()
ax.plot(losses)
ax.set_yscale('log')
plt.show()
plt.savefig("test.png")
raw_data=raw_data.to(traindevice)
for x in raw_data:
    z=model.encoder(x)
    xprime=model.decoder(z)
    #im=Image.fromarray((xprime.detach().numpy()*255).round(0).astype(int),mode="L")
    #im.show()
    #im.save("test.png")
    print(x,z.detach().cpu().numpy().round(2),xprime.detach().cpu().numpy().round(2))

    k=0