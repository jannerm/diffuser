import time
import torch
import pdb

def pass_fn(*args, **kwargs):
    pass

def get_device(model):
    param = list(model.parameters())[0]
    return param.device

def to_device(*xs, device='cuda:0', dtype=torch.float32):
    return [x.to(device).type(dtype) for x in xs]

class GenericTrainer:

    def __init__(self, model, dataloader, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = get_device(model)

    def train(self, n_epochs, log_freq=0, callback_freq=0, callback_fn=pass_fn):

        self.n_epochs = n_epochs

        for epoch in range(n_epochs):

            self.train_epoch(epoch, log_freq)

            # model.train()

            # total_loss = 0

            # t0 = time.time()
            # for i, batch in enumerate(dataloader):
            #     observations, actions, next_observations, rewards = utils.to_device(*batch)
            #     predictions = model(observations, actions, next_observations)

            #     loss = criterion(predictions, rewards)
            #     total_loss += loss.item()

            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            #     if i % args.log_freq == 0:
            #         t_total = time.time() - t0
            #         avg_loss = total_loss / (i + 1)
            #         print((
            #             f'    {os.environ["JOB_PREFIX"]}epoch: {epoch} / {args.n_epochs} | iter {i:4d} / {len(dataloader):4d} | '
            #             f'loss {loss.item():.5f} | loss_avg: {avg_loss:.5f} | '
            #             f't: {t_total:.2f}'
            #         ))
            #         t0 = time.time()

            if callback_freq and (epoch + 1) % callback_freq == 0:
                callback_fn(epoch + 1, self.model)

    def train_epoch(self, epoch=0, log_freq=0):

        self.model.train()
        total_loss = 0

        t0 = time.time()
        for i, batch in enumerate(self.dataloader):
            batch = to_device(*batch, device=self.device)
            inputs = batch[:-1]
            targets = batch[-1]

            predictions = self.model(*inputs)

            loss = self.criterion(predictions, targets)
            import pdb
            pdb.set_trace()
            print(predictions.size())
            print(targets.size())
            print(self.criterion)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if log_freq and i % log_freq == 0:
                t_total = time.time() - t0
                avg_loss = total_loss / (i + 1)
                print(
                    f'    epoch: {epoch} / {self.n_epochs} | '
                    f'iter {i:4d} / {len(self.dataloader):4d} | '
                    f'loss {loss.item():.5f} | loss_avg: {avg_loss:.5f} | '
                    f't: {t_total:.2f}'
                )
                t0 = time.time()
