from tqdm import trange
from utils.utils import *
from sklearn.metrics import precision_score, recall_score, f1_score

SAVE_EVERY = 2


def validate_model(model, val_loader):
    model.eval()
    total_loss = 0

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for embeds, tags, mask in val_loader:
            feats = model(embeds)
            loss = model.loss(feats, tags, mask.byte())  # Calculate loss
            total_loss += loss.item()
            best_paths = model.predict(feats, mask.byte())

            for i in range(len(tags)):
                for j in range(len(tags[i])):
                    if mask[i][j].item():
                        all_predictions.append(best_paths[i][j])
                        all_true_labels.append(tags[i][j])

    avg_loss = total_loss / len(val_loader)
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    return avg_loss, precision, recall, f1


def train_and_validate(model, train_loader, val_loader, optimizer, epochs, saved_filename, saved_dir,
                       loaded_filename=None, loaded_dir=None):
    if loaded_filename is not None and loaded_dir is not None:
        loaded_epoch = load_checkpoint(loaded_filename, loaded_dir, model)
    else:
        loaded_epoch = 0

    for epoch in trange(loaded_epoch + 1, loaded_epoch + 1 + epochs):
        model.train()
        total_loss = 0

        for embeds, tags, mask in train_loader:
            model.zero_grad()

            feats = model(embeds)
            loss = model.loss(feats, tags, mask.byte())
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        val_loss, val_precision, val_recall, val_f1 = validate_model(model, val_loader)
        train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch}/{loaded_epoch + epochs}, "
            f"Training Loss: {train_loss}, "
            f"Val loss: {val_loss}, "
            f"Val precision: {val_precision}, "
            f"Val recall: {val_recall}, "
            f"Val F1: {val_f1}")
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(saved_filename, saved_dir, model, epoch, train_loss, val_loss)
