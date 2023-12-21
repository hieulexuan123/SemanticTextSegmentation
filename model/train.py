from tqdm import trange
import torch
from utils.utils import *
import os


def train(model, optimizer, loss_fn, train_loader, val_loader, params, model_dir, restore_file=None):
    if restore_file is not None:
        restore_path = os.path.join(
            model_dir, restore_file + '.pth.tar')
        load_checkpoint(restore_path, model, optimizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_acc = 0
    for epoch in trange(params['num_epochs']):
        # train
        model.train()
        train_losses = []
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.view(-1).to(device)
            output = model(embeddings)
            train_loss = loss_fn(output, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())
        avg_train_loss = torch.FloatTensor(train_losses).mean().item()

        # val
        with torch.no_grad():
            model.eval()
            val_losses = []
            correct = 0
            total = 0
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.view(-1).to(device)
                output = model(embeddings)
                val_loss = loss_fn(output, labels)
                val_losses.append(val_loss)

                # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
                mask = (labels >= 0)
                output = torch.argmax(output, dim=1)
                correct += (output == labels).sum().item()
                total += mask.sum().item()

            avg_val_loss = torch.FloatTensor(val_losses).mean().item()
            val_acc = correct / total

        print(
            f'Epoch {epoch + 1}/{params["num_epochs"]}: Train_loss = {avg_train_loss}, Val_loss = {avg_val_loss}, Val_acc = {val_acc}')

        is_best = val_acc >= best_val_acc

        # Save weights
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint=model_dir)

        if is_best:
            best_val_acc = val_acc
            print(f'Found new best accuracy: {best_val_acc}')