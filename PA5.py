import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import vocab
import io
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
from torchvision import transforms

import json
import os
import datetime
import warnings
import random

from MyTransformer import MyTransformer
from bleu import get_bleu

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class My_dataset(Dataset):
        # initialize dataset class
    def __init__ (self, dir, dset, vocab) -> object:
        j = json.load(open(f"{dir}{dset}_data.json"))
        self.dataDir = dir
        self.data = j['images']
        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])
        self.vocab = vocab

    def __len__ (self):
        #return length of the total dataset
        return len(self.data)
        
    def __getitem__ (self, idx):
        #return data with index idx
        imagePath = f"{self.dataDir}Images/{self.data[idx]['filename']}"
        image = Image.open(imagePath)
        src = self.transform(image)
        x = 0#random.randint(0, 4)
        tgt = tokenize(self.data[idx]['sentences'][x]['tokens'], self.vocab)
        return (src, torch.tensor(tgt))

def tokenize(sentence, vocab):
    tok_arr = [vocab['<bos>']]
    for token in sentence:
        tok_arr.append(vocab[token])
    tok_arr.append(vocab['<eos>'])
    return tok_arr

def build_vocab(dir):
    counter = Counter()
    for dat in json.load(open(f"{dir}test_data.json"))['images']:
        for sent in dat['sentences']:
            counter.update(sent['tokens'])

    for dat in json.load(open(f"{dir}training_data.json"))['images']:
        for sent in dat['sentences']:
            counter.update(sent['tokens'])

    for dat in json.load(open(f"{dir}val_data.json"))['images']:
        for sent in dat['sentences']:
            counter.update(sent['tokens'])
    counts = list(counter.items())
    counts.sort(key=lambda a: (-a[1], a[0]))
    return vocab(dict(counts), specials=["<unk>", "<pad>", "<bos>", "<eos>"])

def create_batch(each_data_batch,PAD_IDX):
    im_batch, en_batch = torch.zeros(len(each_data_batch), each_data_batch[0][0].shape[0], each_data_batch[0][0].shape[1], each_data_batch[0][0].shape[2]), []
    for i, (im_item, en_item) in enumerate(each_data_batch):
        im_batch[i] = im_item
        en_batch.append(en_item)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return im_batch, en_batch

def PA5_train(dataDir, expDir, _p):
    if(6000/_p["Batch Size"] < _p["Num Batches"]):
        print("ERROR: maximum number of batches exceeded")
        return
    
    if(os.path.exists(expDir+"/params/saved_model.pth")):
        print("Training skipped due to pre-existing model")
        return

    vocab = build_vocab(dataDir)

    train_dataset = My_dataset(dataDir, "training", vocab)
    valid_dataset = My_dataset(dataDir, "val", vocab)
    print(len(valid_dataset))

    create_batch_padded = lambda x: create_batch(x, vocab['<pad>'])

    training_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=_p["Batch Size"],
                                                      shuffle=True,
                                                      collate_fn=create_batch_padded)
    
    validation_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                      batch_size=20,
                                                      shuffle=False,
                                                      collate_fn=create_batch_padded)
   
    #setup initial weights
    model = MyTransformer(_p["Num Encoder Layers"],
                          _p["Num Decoder Layers"],
                          _p["Embedding Size"],
                          _p["Num Heads"],
                          len(vocab),
                          DEVICE,
                          _p["Image Size"],
                          _p["Patch Size"],
                          _p["Num Classes"],
                          _p["Dim Forward"],
                          _p["Dropout"])
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    print(model.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    train_epochLoss = [] #the loss after each epoch in training data
    valid_epochLoss = [] #the loss after each epoch in validation data
    minLoss = 1000.0
    lastImprovement = 0
    for epoch in range(_p["Max Epochs"]):
        print("Epoch", epoch, "starting:", datetime.datetime.now().time())

        #save parameters
        torch.save(model.state_dict(), f"{expDir}/params/translator_e{epoch}.pth")

        train_loss = []
        valid_loss = []

        dat_iter = iter(training_dataloader)
        for b in range(_p["Num Batches"]):
            src, tar = next(dat_iter)
            src = src.to(DEVICE)
            tar = tar.to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, tar[:-1, :])
            loss_val = loss_fn(logits.reshape(-1, logits.shape[-1]), tar[1:, :].reshape(-1))
            loss_val.backward()
            optimizer.step()

            train_loss.append(loss_val.item())
        #print(f"  training finished: {datetime.datetime.now().time()}")

        #validation loss
        val_iter = iter(validation_dataloader)
        for b1 in range(50):
            src, tar = next(val_iter)
            src = src.to(DEVICE)
            tar = tar.to(DEVICE)
            logits = model(src, tar[:-1, :])
            loss_val = loss_fn(logits.reshape(-1, logits.shape[-1]), tar[1:, :].reshape(-1))
            valid_loss.append(loss_val.item())
            #print(f"    validation batch {b1} finished: {datetime.datetime.now().time()}")
        
        #print(f"  validation finished: {datetime.datetime.now().time()}")
        
        #Take the mean of all the mini-batch losses and denote it as your loss of the current epoch
        meanLoss = 0
        for l in train_loss:
            meanLoss += l
        meanLoss = meanLoss/len(train_loss)

        print(f"  Training loss for epoch {epoch}: {meanLoss}")

        #Collect loss for each epoch and save the parameter Theta after each epoch
        train_epochLoss.append(meanLoss)

        #Take the mean of all the mini-batch losses and denote it as your loss of the current epoch
        meanLoss = 0
        for l in valid_loss:
            meanLoss += l
        meanLoss = meanLoss/len(valid_loss)

        print(f"  Validation loss for epoch {epoch}: {meanLoss}")

        #Collect loss for each epoch and save the parameter Theta after each epoch
        valid_epochLoss.append(meanLoss)

        #check if there is no significant change in loss over last 20 iterations
        if (loss_val.item() < minLoss):
            minLoss = loss_val.item()
            lastImprovement = epoch
        if (epoch - lastImprovement > 9):
            print(f"Overfitting detected, stopping at epoch {epoch}. Last improvement seen at {lastImprovement}")
            break
    
    #save classifier
    torch.save(model.state_dict(), expDir+"/params/saved_model.pth")

    #plot loss
    plt.figure()
    plt.plot(range(1, len(train_epochLoss)+1), train_epochLoss, label='Training Loss')
    plt.plot(range(1, len(valid_epochLoss)+1), valid_epochLoss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.xticks(np.arange(0, len(train_epochLoss)+1, 5))
    ax = plt.gca()
    #ax.set_ylim([0.0, 10.0])
    ax.legend()
    plt.savefig(f"{expDir}/graphs/epochLoss.png")

def enterpretOutput(output):
    ret = torch.zeros(output.shape[0])
    for i, o in enumerate(output):
        ret[i] = (torch.argmax(o))
    return ret

def unTokenize(sent, vocab):
    ret = []
    for s in sent:
        if (s == vocab["<pad>"] or s == vocab["<bos>"] or s == vocab["<eos>"]):
            continue
        ret.append(vocab.lookup_token(s))
    return ret


def scoreEpoch(params, dataset, v, _p):
    m = MyTransformer(_p["Num Encoder Layers"],
                          _p["Num Decoder Layers"],
                          _p["Embedding Size"],
                          _p["Num Heads"],
                          len(v),
                          DEVICE,
                          _p["Image Size"],
                          _p["Patch Size"],
                          _p["Num Classes"],
                          _p["Dim Forward"],
                          _p["Dropout"])
    m.load_state_dict(params)

    if (len(dataset) % 20 != 0):
        print(f"Dataset length {len(dataset)} is not divisible by 20")
        return None

    create_batch_padded = lambda x: create_batch(x, v['<pad>'])

    dl = torch.utils.data.DataLoader(dataset,
                                    batch_size=20,
                                    shuffle=False,
                                    collate_fn=create_batch_padded)

    bleu = 0.0
    data_iterator = iter(dl)
    for b in range(int(len(dataset)/20)):
        src, tar = next(data_iterator)
        src = src.to(DEVICE)
        tar = tar.to(DEVICE)
        logits = m(src, tar[:-1, :])
        for l, t in zip(logits, tar[1:, :]):
            gt = unTokenize(t, v)
            o = unTokenize(enterpretOutput(l), v)
            bleu += get_bleu(o, gt)
    print(bleu)
    bleu /= len(dataset)

    return bleu

def scoreModel(prevParams, finalParams, dataset, v, d):
    if prevParams == None:
        return [scoreEpoch(finalParams, dataset, v, d)]
    ret = []
    for i, p in enumerate(prevParams):
        print("Epoch", i, "scoring:", datetime.datetime.now().time())
        ret.append(scoreEpoch(p, dataset, v, d))
    print("Epoch", len(prevParams), "scoring:", datetime.datetime.now().time())
    ret.append(scoreEpoch(finalParams, dataset, v, d))
    return ret

def PA5_test(dataDir, expDir, params):

    epochs = params["Max Epochs"]

    vocab = build_vocab(dataDir)

    dataset = My_dataset(dataDir, "test", vocab)

    print(dataset[0][0].shape)

    #Load the model
    model = MyTransformer(params["Num Encoder Layers"],
                          params["Num Decoder Layers"],
                          params["Embedding Size"],
                          params["Num Heads"],
                          len(vocab),
                          DEVICE,
                          params["Image Size"],
                          params["Patch Size"],
                          params["Num Classes"],
                          params["Dim Forward"],
                          params["Dropout"])
    finalParams = torch.load(expDir+"/params/saved_model.pth")
    model.load_state_dict(finalParams)

    #load previous models
    epochModels = []
    if (os.path.exists(f"{expDir}/params/translator_e0.pth")):
        for i in range(epochs):
            if (not os.path.exists(f"{expDir}/params/translator_e{i}.pth")):
                epochs = i+1
                break
            epochModels.append(torch.load(f"{expDir}/params/translator_e{i}.pth"))
    else:
        epochModels = None

    # Read the data from the testing data
    dataset = My_dataset(dataDir, "test", vocab)

    src, tar = dataset[0]
    src = src.unsqueeze(0).to(DEVICE)
    tar = tar.unsqueeze(1).to(DEVICE)
    groundTruth = unTokenize(tar[1:, :], vocab)
    print(f"ground truth: {groundTruth}")

    logits = model(src, tar[:-1, :])
    output = unTokenize(enterpretOutput(logits), vocab)
    print(enterpretOutput(logits))
    print(f"model output: {output}")

    print(get_bleu(output, groundTruth))

    testScore = scoreModel(epochModels, finalParams, dataset, vocab, params)

    bestScore = 0.0
    bestIndex = 0
    for i, s in enumerate(testScore):
        if (s > bestScore):
            bestScore = s
            bestIndex = i

    if (bestIndex == len(testScore)-1):
        print(f"The best model is saved_model with a socre of {bestScore}")
    else:
        print(f"The best model is translator_e{bestIndex} with a score of {bestScore}")

    if (epochModels != None):

        #plot testing score per epoch
        plt.figure()
        plt.plot(range(1, len(testScore)+1), testScore)
        plt.xlabel("Epoch")
        plt.ylabel("BLEU Score")
        plt.title(f"BLEU Score by epoch")
        ax = plt.gca()
        ax.set_ylim([0.0, 100.0])
        plt.savefig(f"{expDir}/graphs/bs.png")

    with open(f"{expDir}/accuracy.txt", "w") as fd:
        if (bestIndex == len(testScore)-1):
            fd.write(f"The best model is saved_model with a socre of {bestScore}")
        else:
            fd.write(f"The best model is translator_e{bestIndex} with a score of {bestScore}")

if __name__ == "__main__":
    print(DEVICE)
    print(torch.__version__)
    torch.set_default_dtype(torch.float64)
    dataDir = "data/"
    expDir = "Experiment0"

    params = {}
    params["Batch Size"] = 30
    params["Num Batches"] = 10
    params["Max Epochs"] = 10

    params["Num Encoder Layers"] = 3
    params["Num Decoder Layers"] = 3
    params["Embedding Size"] = 512
    params["Num Heads"] = 8
    params["Image Size"] = 256
    params["Patch Size"] = 16
    params["Num Classes"] = 1000
    params["Dim Forward"] = 512
    params["Dropout"] = 0.1

    print(expDir, params)
    os.makedirs(expDir, exist_ok=True)
    os.makedirs(f"{expDir}/graphs", exist_ok=True)
    os.makedirs(f"{expDir}/params", exist_ok=True)
    filehandler = open(f"{expDir}/params.json", "w")
    json.dump(params, filehandler)
    filehandler.close()

    warnings.filterwarnings("ignore")

    PA5_train(dataDir, expDir, params)
    PA5_test(dataDir, expDir, params)