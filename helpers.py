import numpy as np
import torch
from tqdm import tqdm


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) /\
                float(len(set_true | set_pred))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(epoch, device, model, optimizer, training_loader, scaler):
    model.train()
    fin_targets = []
    fin_outputs = []
    running_loss = 0.0

    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss and store targets/outputs for metrics
        running_loss += loss.item()
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(
            outputs).cpu().detach().numpy().tolist())

    # Calculate average loss for the epoch
    avg_loss = running_loss / len(training_loader)

    # Return the average loss, and the collected outputs and targets
    return avg_loss, fin_outputs, fin_targets


def validation(testing_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(
                outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def predict_tags(poem_text, model, tokenizer, max_len, device, tags, threshold=0.5):
    inputs = tokenizer.encode_plus(
        poem_text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True
    )

    input_ids = torch.tensor(
        inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(
        inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(
        inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask, token_type_ids)

    sigmoid_output = torch.sigmoid(output)

    sigmoid_output = sigmoid_output.cpu().numpy().flatten()

    predicted_tags = [tags[i] for i in range(
        len(sigmoid_output)) if sigmoid_output[i] >= threshold]

    return predicted_tags
