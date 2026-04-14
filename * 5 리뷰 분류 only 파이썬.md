## Classifying movie reviews: a binary classification problem

> ### The IMDB dataset

* A set of 50,000 highly polarized reviews (positive and negative) from the Internet Movie Database
* 25,000 reviews for training and 25,000 reviews for testing
* Each set consists of 50% negative and 50% positive reviews
* Why use separate training and test sets?

* Loading the IMDB dataset

import re, random
import torch
from collections import Counter
from datasets import load_dataset

ds = load_dataset("imdb")

id2label = {0: "neg", 1: "pos"}
train_list = [(id2label[int(r["label"])], r["text"]) for r in ds["train"]]
test_list  = [(id2label[int(r["label"])], r["text"]) for r in ds["test"]]
print(train_list)

train_list[1000]

* `train_list` and `test_list` are lists of reviews
  * Each review is a string.
  
* Originally, the label is 0 or 1, where 0 stands for *negative* and 1 stands for *positive*.
  

* Let's build vocab from training set.
* We need the vocab to convert the string (i.e., review) into a list of indices.

# Simple tokenizer (English)
def simple_tokenize(s: str):
    # alphanumeric word tokens, lowercased
    return re.findall(r"\b\w+\b", s.lower())

# Build vocab from training set
#    Keep top-N frequent tokens
MAX_VOCAB = 10000
specials = ["<unk>", "<pad>"] # unk -> 가지고 있는 무비
                              # pad -> 각 리뷰는 길이가 다름 그래서 짧은 애들한테 의미없는걸 붙여서 길이를 맞춤

counter = Counter()
for _, txt in train_list:
    counter.update(simple_tokenize(txt))

most_common = [w for w, _ in counter.most_common(MAX_VOCAB - len(specials))]
itos = specials + most_common
stoi = {w: i for i, w in enumerate(itos)}
UNK_IDX = stoi["<unk>"]
PAD_IDX = stoi["<pad>"]

print(stoi)

* Build data pipelines: strings → numbers (specifically, indices)



label_to_int = {"neg": 0, "pos": 1}

def text_pipeline(x: str):
    ids = [stoi.get(tok, UNK_IDX) for tok in simple_tokenize(x)]
    return torch.tensor(ids, dtype=torch.long)

def label_pipeline(y: str):
    return torch.tensor(label_to_int[y], dtype=torch.float32)

print(train_list[0][1])
print(text_pipeline(train_list[0][1]))

* Train/valid split (90/10)

random.seed(42)
random.shuffle(train_list)
split_idx = int(len(train_list) * 0.9)
train_data = train_list[:split_idx]
valid_data = train_list[split_idx:]
test_data  = test_list

print(f"Vocab size: {len(stoi)}, Train/Valid/Test: {len(train_data)}/{len(valid_data)}/{len(test_data)}")

> ### Batching with collate function

* We define a `collate_batch` that:
    1. tokenizes and numericalizes
    2. pads sequences with the `<pad` index, and
    3. stacks labels as float tensors (for `BCEWithLogitsLoss`)
* We then build PyTorch `DataLoader`s for the train/valid/test sets.

import os, random, math, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    text_list, label_list = [], []
    for (label, text) in batch:
        text_list.append(text_pipeline(text))
        label_list.append(label_pipeline(label))
    # pad to the same length
    text_padded = pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(label_list)
    return text_padded.to(DEVICE), labels.to(DEVICE)

BATCH_SIZE = 128
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
len(train_loader), len(valid_loader), len(test_loader)

for (x,y) in train_loader:
    print(x.shape)

x.shape

x[0]



> ### Building the network

* Note that the input data consists of lists of indices (i.e., tokens), and the labels are scalars.
* First, we map tokens to vectors with an `Embedding` layer.
    * This is a fundamental building block

* Note that the input data consists of lists of indices (i.e., tokens), and the labels are scalars.  
* First, we map tokens to vectors with an `Embedding` layer.  
    * This is a fundamental building block for natural language models, since raw token IDs carry no semantic meaning.  
    * Each token index is mapped to a dense vector of fixed dimensionality (e.g., 128).  
    * These vectors are trainable parameters — during training, the embedding matrix is updated so that semantically similar words tend to have similar vectors.  
    * In our pipeline, the embedding layer produces a `(batch_size, sequence_length, embed_dim)` tensor (in this example, it will later be pooled into a fixed-size sentence representation).  


* Consider a simple stack of fully connected (`Linear`) layers with `relu` activations.
  * `nn.Linear(in_dim, h)`
  * The arguments `in_dim` and `h` represent the number of input nodes and hidden nodes, respectively.
  * *The number of hidden nodes* is a dimension in the representation space of the layer.
  * Recall that `output = dot(W, input) + b`.
  * Then, what is the shape of the weight matrix `W`?

* The dimensionality of the representation space = How much freedom you are allowing the network to have when learning internal representations

* There are two key architecture decisions about a stack of `Linear` layers:
  * How many layers to use
  * How many hidden units to choose for each layer
  
* In this example, we will use:
  * Two hidden layers with 32 hidden units each
  * A third layer that will output the scalar prediction regarding the sentiment of the input review

* The intermediate layers will use `relu` as their activation function, and the final layer will output a single real value (it can be converted into a probability by applying the sigmoid function).
  * A `relu` (rectified linear unit) is a function meant to zero out negative values,
  $f(x)=x^+=max(0,x)$

    <img src="https://drive.google.com/uc?id=1BSU0hmpQz_gnKHwczouNrn_adRkBqLtm" width="400">

  * A sigmoid squashes arbitrary values into the `[0, 1]` interval, outputting something that can be interpreted as a probability,
  $f(x)=1/({1+\exp(-x)})$

    <img src="https://drive.google.com/uc?id=14QEmx2LHCtz1qUaMwVAdk9nbLbl52PZS" width="400">
    

class SentimentMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dims=(32, 32), pad_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx if pad_idx is not None else 0)
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]  # output logit
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, E)
        # Create mask for non-pad tokens
        if hasattr(self.embedding, "padding_idx") and self.embedding.padding_idx is not None:
            pad_idx = self.embedding.padding_idx
        else:
            pad_idx = 0
        mask = (x != pad_idx).unsqueeze(-1)  # (B, T, 1)
        emb = emb * mask  # zero out pad embeddings
        lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_pooled = emb.sum(dim=1) / lengths  # (B, E)
        logits = self.mlp(mean_pooled).squeeze(1)  # (B,)
        return logits

model = SentimentMLP(vocab_size=len(stoi), embed_dim=64, hidden_dims=(32, 32), pad_idx=PAD_IDX).to(DEVICE)
model

* Why are the activation functions necessary?

* Finally, we need to choose a loss function and an optimizer.
  * We are dealing with a binary classification problem and the output of our network is a probability. --> `binary_crossentropy` loss
    * `binary_crossentropy(y_pred, y) = -(y*log(y_pred) + (1-y)*log(1-y_pred))`

    ```python
      def binary_crossentropy(y_pred, y):
            if y == 1:
                return -log(y_pred)
            else:
                return -log(1 - y_pred)
    ```
    
  * Note that it is not the only viable choice. For example, `mean_squared_error`.
  * *cross entropy* measures the distance between probability distributions or, in this case, between the ground-truth distribution and the predictions.
  

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

> ### Training and validation loops

* We implement explicit train/eval loops.
* During training we zero gradients, backproprgate, and take an optimizer step.
* Accuracy is computed with `sigmoid(logits) >= 0.5`.

def binary_accuracy_from_logits(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == targets).sum().item()
    return correct / targets.numel()

def run_epoch(dataloader, model, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_acc, total_count = 0.0, 0.0, 0
    for xb, yb in dataloader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            optimizer.step()
        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_acc  += binary_accuracy_from_logits(logits.detach(), yb) * bs
        total_count += bs
    return total_loss / total_count, total_acc / total_count

EPOCHS = 20
history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
best_val_acc, best_state = 0.0, None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, model, criterion, optimizer)
    va_loss, va_acc = run_epoch(valid_loader, model, criterion, optimizer=None)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["valid_loss"].append(va_loss)
    history["valid_acc"].append(va_acc)

    print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | valid_loss={va_loss:.4f} acc={va_acc:.4f}")
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)


* Training the model for 20 epochs in mini-batches of 128 samples.
* At the same time, we will monitor loss and accuracy on the 2,500 samples in a validation set.

* We can plot the training and validation loss (or accuracy).

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["valid_loss"], label="valid_loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss Curves")
plt.show()

plt.figure()
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["valid_acc"], label="valid_acc")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy Curves")
plt.show()


* The training loss decreases with every epoch, and the training accuracy increases with every epoch.
  * This is what we expect when running gradient-descent optimization.
* But that isn't the case for the validation loss and accuracy.
  * They seem to peak at the early epoch.
* Important note: a model that performs better on the training data isn't necessarily a model that will do better on data it has never seen before.
  * *overfitting*
  * The network is quickly overoptimized on the training data, and it learned representations that are specific to the training data.
  * These representations don't generalize to data outside of the training set.
  
* To prevent overfitting, we could stop training after a specific epoch.
  * We will learn various techniques to mitigate overfitting later.
  

> ### Prediction on new data with a trained network

* We restore the best validation checkpoint (if any) and evaluate on the held-out test set.

print(type(best_state))

test_loss, test_acc = run_epoch(test_loader, model, criterion, optimizer=None)
print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f}")


> ### Inference examples

* We create a convenience function that predicts the positive sentiment probability for a given sentence by passing it
through the same preprocessing pipeline and the trained MLP.

def predict_sentiment(model, sentence: str):
    model.eval()
    with torch.no_grad():
        x = text_pipeline(sentence).unsqueeze(0).to(DEVICE)  # (1, T)
        logit = model(x)
        prob = torch.sigmoid(logit).item()
        return {"prob_pos": prob, "label": "pos" if prob >= 0.5 else "neg"}

examples = [
    "This movie was absolutely fantastic. I loved every minute.",
    "Terrible plot and wooden acting. I want my time back."
]
for s in examples:
    print(s, "->", predict_sentiment(model, s))

> ### Saving the trained model

* We save the learned parameters using `state_dict`. You can later reload them with `load_state_dict` on the same model definition.

save_path = "imdb_sentiment_mlp_state_dict.pt"
torch.save(model.state_dict(), save_path)
save_path


!ls -al

!pwd

* A Colab instance runs on Google Cloud, which means that any files saved in the local runtime storage will be lost once the session disconnects or the runtime is reset.  
* To preserve your saved files (e.g., trained model weights), you should **mount your Google Drive** and save the files directly to your personal cloud storage.

from google.colab import drive
drive.mount('/content/gdrive')

%cd /content/gdrive

!ls

%cd MyDrive/exp

save_path = "imdb_sentiment_mlp_state_dict.pt"
torch.save(model.state_dict(), save_path)
save_path

> ### Adding regularization

* Let's consider L2 regularization.
    * Note that L2 regularization is also known as weight decay (Strictly speaking, they are **not exactly the same under all optimizers**).

# create a new model
model = SentimentMLP(vocab_size=len(stoi), embed_dim=64, hidden_dims=(32, 32), pad_idx=PAD_IDX).to(DEVICE)

# set weight decay argument
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# run again
EPOCHS = 20
history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
best_val_acc, best_state = 0.0, None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, model, criterion, optimizer)
    va_loss, va_acc = run_epoch(valid_loader, model, criterion, optimizer=None)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["valid_loss"].append(va_loss)
    history["valid_acc"].append(va_acc)

    print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | valid_loss={va_loss:.4f} acc={va_acc:.4f}")
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["valid_loss"], label="valid_loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss Curves")
plt.show()

plt.figure()
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["valid_acc"], label="valid_acc")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy Curves")
plt.show()

* Let's consider **dropout**.
    * *Dropout* is one of the most effective and most commonly used regularization techniques for neural networks.

    * It consists of randomly dropping out (setting to zero) a number of output features of the layer during training.
    * E.g., [0.2, 0.5, 1.3, 0.8, 1.1] --> (dropout) --> [0, 0.5, 1.3, 0, 1.1]
  
    * The *dropout rate* is the fraction of the features that are zeroed out (~= probability of an element to be zeroed).

    * At test time, no units are dropped out.
  * Instead, the layer's output values are scaled down by a factor equal to *(1-the dropout rate)* to balance for the fact that more units are active than at training time.
  ><img src="https://drive.google.com/uc?id=1nfP0HxqbBcW-isMbCDuDkD-Bu_x7w65_" width="800">
  
* Implementation using Numpy

  ```python
    # At training time, we zero out 50% of activations.
    layer_output *= np.random.randint(0, high=2, size=layer_output.shape)

    # At test time, we scale down the output.
    layer_output *= 0.5
  ```

* Another implementation (in practice)

  ```python
    # At training time
    layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
    layer_output /= 0.5 # divided by (1-p)
  ```

* In PyTorch,

  ```python
    torch.nn.Dropout(p=0.5)
  ```

* Adding dropout to the IMDB network






class SentimentMLPwithDropout(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dims=(32, 32), dropout_p=0.5, pad_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx if pad_idx is not None else 0)
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(p=dropout_p)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]  # output logit
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)  # (B, T, E)
        # Create mask for non-pad tokens
        if hasattr(self.embedding, "padding_idx") and self.embedding.padding_idx is not None:
            pad_idx = self.embedding.padding_idx
        else:
            pad_idx = 0
        mask = (x != pad_idx).unsqueeze(-1)  # (B, T, 1)
        emb = emb * mask  # zero out pad embeddings
        lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_pooled = emb.sum(dim=1) / lengths  # (B, E)
        logits = self.mlp(mean_pooled).squeeze(1)  # (B,)
        return logits

model = SentimentMLPwithDropout(vocab_size=len(stoi), embed_dim=64, hidden_dims=(32, 32), dropout_p=0.7, pad_idx=PAD_IDX).to(DEVICE)
model

# no weight decay
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# run again
EPOCHS = 20
history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
best_val_acc, best_state = 0.0, None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, model, criterion, optimizer)
    va_loss, va_acc = run_epoch(valid_loader, model, criterion, optimizer=None)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["valid_loss"].append(va_loss)
    history["valid_acc"].append(va_acc)

    print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | valid_loss={va_loss:.4f} acc={va_acc:.4f}")
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["valid_loss"], label="valid_loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss Curves")
plt.show()

plt.figure()
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["valid_acc"], label="valid_acc")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy Curves")
plt.show()

> ### Further experiments

* Try using different number of hidden layers.
* Try using layers with more hidden units or fewer hidden units.
* Try using the `mse` loss function instead of `binary_crossentropy`
* Try using other activation functions (e.g. `tanh`, `sigmoid`) instead of `relu`
* Try other optimizers instead of `Adam`
