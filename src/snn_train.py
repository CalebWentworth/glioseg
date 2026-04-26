from tqdm.auto import tqdm
import torchio as tio
from torch.utils.data import DataLoader
from snntorch import spikegen
import torch
import time

class TrainNetwork:
    
    def __init__(self,
                 train_queue,
                 val_queue,
                 model,
                 optimizer,
                 loss_fn,
                 device=None,
                 spike_len=8,
                 lr=5e-4,
                 batch_size=1,
                 num_workers=0,
                 num_epochs=5
                 ):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_queue = train_queue
        self.val_queue = val_queue
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn=loss_fn
        self.spike_len = spike_len
        self.lr = lr
        self.batch_size = batch_size
        self.numworkers = num_workers
        self.num_epochs = num_epochs

    def train(self):
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        start = time.time()

        training_loader = DataLoader(
            self.train_queue, batch_size=self.batch_size, num_workers=self.numworkers
        )

        pbar = tqdm(training_loader, desc="Training", leave=True, dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar, start=1):
            t1c = batch["t1c"][tio.DATA]
            t1n = batch["t1n"][tio.DATA]
            t2w = batch["t2w"][tio.DATA]
            t2f = batch["t2f"][tio.DATA]

            data = torch.cat([t1c, t1n, t2w, t2f], dim=1).to(self.device, dtype=torch.float32)
            targets = batch["seg"][tio.DATA].to(self.device).long()

            spike_train = spikegen.rate(data, num_steps=self.spike_len)

            self.model.init_mem()
            mem_rec = []
            for step in range(self.spike_len):
                _, mem_out = self.model(spike_train[step])
                mem_rec.append(mem_out)

            logits = torch.stack(mem_rec).mean(dim=0)
            loss = self.loss_fn(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bsz = data.size(0)
            running_loss += loss.item() * bsz
            total_samples += bsz

            avg_loss = running_loss / max(total_samples, 1)
            pbar.set_postfix(batch=batch_idx, avg_loss=f"{avg_loss:.4f}", refresh=True)

        elapsed = time.time() - start

        return {
            "loss": running_loss / max(total_samples, 1),
            "samples": total_samples,
            "batches": len(training_loader),
            "seconds": elapsed,
            "samples_per_sec": total_samples / max(elapsed, 1e-8),
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        start = time.time()

        val_loader = DataLoader(
            self.val_queue, batch_size=self.batch_size, num_workers=self.numworkers
        )

        pbar = tqdm(val_loader, desc="Validation", leave=True, dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar, start=1):
            t1c = batch["t1c"][tio.DATA]
            t1n = batch["t1n"][tio.DATA]
            t2w = batch["t2w"][tio.DATA]
            t2f = batch["t2f"][tio.DATA]

            data = torch.cat([t1c, t1n, t2w, t2f], dim=1).to(self.device, dtype=torch.float32)
            targets = batch["seg"][tio.DATA].to(self.device).long()

            spike_train = spikegen.rate(data, num_steps=self.spike_len)

            self.model.init_mem()
            mem_rec = []
            for step in range(self.spike_len):
                _, mem_out = self.model(spike_train[step])
                mem_rec.append(mem_out)

            logits = torch.stack(mem_rec).mean(dim=0)
            loss = self.loss_fn(logits, targets)

            bsz = data.size(0)
            running_loss += loss.item() * bsz
            total_samples += bsz

            avg_loss = running_loss / max(total_samples, 1)
            pbar.set_postfix(batch=batch_idx, avg_loss=f"{avg_loss:.4f}", refresh=True)

        elapsed = time.time() - start
        
        return {
            "loss": running_loss / max(total_samples, 1),
            "samples": total_samples,
            "batches": len(val_loader),
            "seconds": elapsed,
            "samples_per_sec": total_samples / max(elapsed, 1e-8),
        }

        