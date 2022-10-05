from pathlib import Path
from time import time
import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader

dim = 64

class Net(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)

class e2eGNN:
    def __init__(self, dataset, model_dir, log):
        self.model_dir = Path(model_dir)
        self.log = log

        # Normalize targets to mean = 0 and std = 1.
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        self.mean, self.std = mean.item(), std.item()
        
        # Split datasets.
        test_dataset = dataset[:10000]
        val_dataset = dataset[10000:20000]
        train_dataset = dataset[20000:]
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        self.loaders = {
            'test': test_loader,
            'val': val_loader,
            'train': train_loader,
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net(dataset.num_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                            factor=0.7, patience=5,
                                                            min_lr=0.00001)

    def test(self, loader):
        model = self.model
        device = self.device

        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            error += (model(data) * self.std - data.y * self.std).abs().sum().item()  # MAE
        return error / len(loader.dataset)

    def run_epoch(self, epoch):
        device = self.device
        model = self.model
        train_loader = self.train_loader
        optimizer = self.optimizer

        model.train()
        loss_all = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(data), data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
            
        torch.save(self.model.state_dict(), self.model_dir / 'last_model.pth')
        return loss_all / len(train_loader.dataset)

    def train(self, max_epoch):
        scheduler = self.scheduler

        best_val_error = None
        for epoch in range(1, max_epoch):
            lr = scheduler.optimizer.param_groups[0]['lr']
            init = time.time()
            loss = self.run_epoch(epoch)
            final = time.time() - init
            val_error = self.test(self.loaders['val'])
            scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                torch.save(self.model.state_dict(), self.model_dir / 'best_model.pth')
                test_error = self.test(self.loaders['test'])
                best_val_error = val_error


            epoch_results = {
                'epoch': epoch,
                'lr': lr,
                'loss': loss,
                'validation_error': val_error,
                'test_error': test_error,
                'time': time.strftime("%H:%M:%S", time.gmtime(final)),
            }

            self.log.register(epoch_results)