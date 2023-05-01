import gc
import torch


def validate(model, device, val_loader):
    model.eval()
    val_set_size = len(val_loader.dataset)
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            x, target = data['input_ids'].to(device), data['labels'].to(device)
            loss, logits = model(x,
                                 labels=target,
                                 return_dict=False)

            val_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= val_set_size
    accuracy = correct / val_set_size

    del loss
    del logits
    gc.collect()

    print(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{val_set_size} "
          f"({100. * correct / val_set_size:.0f}%)\n")
    return val_loss, accuracy