import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Hamming score between true and predicted labels for
    multi-label classification.

    Parameters:
    ----------
    y_true : np.ndarray
        The true labels in a binary format (one-hot encoded).
    y_pred : np.ndarray
        The predicted labels in a binary format (probabilities converted to
        binary using a threshold).

    Returns:
    -------
    float
        The Hamming score between the true and predicted labels.
    """

    return np.mean([
        len(set(np.where(y_true[i])[0]).intersection(
            set(np.where(y_pred[i])[0]))) /
        len(set(np.where(y_true[i])[0]).union(set(np.where(y_pred[i])[0])))

        if len(set(np.where(y_true[i])[0])) > 0 else 1
        for i in range(y_true.shape[0])
    ])


def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the binary cross-entropy loss with logits for multi-label
    classification.

    Parameters:
    ----------
    outputs : torch.Tensor
        The model outputs (logits) for multi-label classification.
    targets : torch.Tensor
        The true labels for multi-label classification.

    Returns:
    -------
    torch.Tensor
        The computed binary cross-entropy loss.
    """

    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(device: torch.device, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          training_loader: torch.utils.data.DataLoader,
          scaler: torch.cuda.amp.GradScaler) -> tuple:
    """
    Trains the model for one epoch.

    Parameters:
    ----------
    device : torch.device
        The device (CPU or GPU) to run the model on.
    model : torch.nn.Module
        The PyTorch model to train.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    training_loader : torch.utils.data.DataLoader
        The DataLoader that provides the training data.
    scaler : torch.cuda.amp.GradScaler
        The GradScaler used for mixed-precision training.

    Returns:
    -------
    tuple
        The average loss for the epoch, the predicted outputs,
        and the true targets.
    """
    model.train()
    fin_targets, fin_outputs = [], []
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

        running_loss += loss.item()

        fin_targets.extend(targets.cpu().detach().numpy())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy())

    avg_loss = running_loss / len(training_loader)

    return avg_loss, fin_outputs, fin_targets


def validation(testing_loader: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               device: torch.device) -> tuple:
    """
    Evaluates the model on the validation or test dataset.

    Parameters:
    ----------
    testing_loader : torch.utils.data.DataLoader
        The DataLoader that provides the test/validation data.
    model : torch.nn.Module
        The PyTorch model to evaluate.
    device : torch.device
        The device (CPU or GPU) to run the model on.

    Returns:
    -------
    tuple
        The predicted outputs and true targets for the evaluation data.
    """
    model.eval()
    fin_targets, fin_outputs = [], []

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


def predict_tags(poem_text: str, model: torch.nn.Module,
                 tokenizer: torch.nn.Module, max_len: int,
                 device: torch.device, tags: list,
                 threshold: float = 0.5) -> list:
    """
    Predicts the tags for a given poem text using the trained model.

    Parameters:
    ----------
    poem_text : str
        The poem text for which to predict tags.
    model : torch.nn.Module
        The trained PyTorch model.
    tokenizer : torch.nn.Module
        The tokenizer used to process the poem text.
    max_len : int
        The maximum length for tokenized input sequences.
    device : torch.device
        The device (CPU or GPU) to run the model on.
    tags : list
        The list of tags corresponding to the output of the model.
    threshold : float, optional
        The threshold for predicting a tag (default is 0.5).

    Returns:
    -------
    list
        A list of predicted tags for the given poem text.
    """

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

    sigmoid_output = torch.sigmoid(output).cpu().numpy().flatten()

    return [tags[i] for i in range(
        len(sigmoid_output)) if sigmoid_output[i] >= threshold]


def plot_loss_score(train_losses: list, val_losses: list,
                    train_hamming_scores: list,
                    val_hamming_scores: list) -> None:
    """
    Plots the training and validation loss and Hamming scores over epochs.

    Parameters:
    ----------
    train_losses : list
        The list of training losses for each epoch.
    val_losses : list
        The list of validation losses for each epoch.
    train_hamming_scores : list
        The list of training Hamming scores for each epoch.
    val_hamming_scores : list
        The list of validation Hamming scores for each epoch.
    """

    plt.figure(figsize=(12, 6))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Hamming Loss')
    plt.ylim(0.25, 0.9)
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.xticks(range(len(train_losses)))

    # Hamming Score Curve
    plt.subplot(1, 2, 2)
    plt.plot(train_hamming_scores, label='Training Hamming Score')
    plt.plot(val_hamming_scores, label='Validation Hamming Score')
    plt.xlabel('Epoch')
    plt.ylabel('Hamming Score')
    plt.title('Training vs Validation Hamming Score')
    plt.legend()
    plt.xticks(range(len(train_hamming_scores)))

    plt.tight_layout()
    plt.show()
