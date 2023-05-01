import gc


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_set_size = len(train_loader.dataset)
    num_batches = len(train_loader)
    train_loss = 0.0
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        x, target = data['input_ids'].to(device), data['labels'].to(device)
        optimizer.zero_grad()
        loss, logits = model(x,
                             labels=target,
                             return_dict=False)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 10 == 0:
            batch_size = len(x)
            print(f"Train Epoch: {epoch} [{batch_idx * batch_size}/{train_set_size} "
                  f"({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.6f}")

        del loss
        del logits
        gc.collect()
    accuracy = correct /train_set_size
    print(f'Train Accuracy after epoch {epoch}: {accuracy}')
    avg_train_loss = train_loss / num_batches

    return avg_train_loss, accuracy