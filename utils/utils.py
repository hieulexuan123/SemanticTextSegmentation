import os
import torch


def load_checkpoint(filename, dir_path, model):
    file_path = os.path.join(dir_path, filename)
    if not os.path.exists(file_path):
        raise f"{file_path} does not exist"

    print("loading %s" % filename)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    print(f"epoch {epoch}, train_loss={train_loss}, val_loss={val_loss}")
    return epoch


def save_checkpoint(filename, dir_path, model, epoch, train_loss, val_loss):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    checkpoint = {"state_dict": model.state_dict(), "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
    file_path = os.path.join(dir_path, filename + ".epoch%d" % epoch)
    torch.save(checkpoint, file_path)
    print("saved %s" % filename)
