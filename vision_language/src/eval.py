import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.datasets.common import maybe_dictionarize, get_dataloader
from src.datasets.registry import get_dataset
from src.modeling import get_model

class IndexWrapper(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
    def __getitem__(self, index):
        instance = self.dataset[index]
        if isinstance(instance, dict):
            instance["index"] = index
            return instance
        return *instance, index
    
    def __len__(self):
        return len(self.dataset)
    
def eval_single_dataset(model, dataloader, args, train=False):
    update = False
    if type(dataloader) == str:
        dataset = get_dataset(
            dataloader,
            model.encoder.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        model.update_dataset(dataloader)

        if train:
            dataloader = get_dataloader(dataset, train, args)
        else:
            dataloader = dataset.test_loader
            
        dataloader = args.accelerator.prepare(dataloader)
        update = True
    
    model.eval()    
    with torch.inference_mode():
        top1, correct = 0.0, 0.0
        preds, labels, losses = [], [], []
        for _, data in enumerate(tqdm(dataloader, disable=args.no_tqdm)):
            data = maybe_dictionarize(data)
            logits = model(data["images"])
                
            y = data["labels"]
            
            preds.append(F.softmax(logits, dim=1).cpu())
            labels.append(y)
            losses.append(F.cross_entropy(logits, y, reduction='none').detach())
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            
        preds, labels, losses = torch.cat(preds), torch.cat(labels), torch.cat(losses)
        top1 = correct / len(labels)

    print(f"Done evaluating. Accuracy: {100*top1:.2f}%")
    if update:
        model.update_dataset(args.train_dataset)
    return {"top1": top1, "preds":preds, "labels": labels, "loss": losses, "dataloader":dataloader}


def get_n_shots(dataset, shots, n_class, args):
    path = f"{args.save}/{args.train_dataset}/{args.data_ratio}_shots_{args.seed}.pt"
    if os.path.isfile(path):
        return torch.load(path)
    
    index_dataset = IndexWrapper(dataset)
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    targets = - torch.ones(len(dataset), dtype=torch.long)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, disable=args.no_tqdm)):
            batch = maybe_dictionarize(batch, index=True)
            targets[batch["index"]] = batch["labels"].to(targets.device)
            if i >= 1000:
                print("Too much data, breaking ...")
                break

    to_keep = torch.tensor([], dtype=torch.long)
    for c in range(n_class):
        cond = (targets == c)
        ids_c = torch.arange(len(targets))[cond]
        a = torch.randperm(len(ids_c))
        to_keep = torch.cat((to_keep, ids_c[a[-shots:]]))
        
    os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    torch.save(to_keep, path)
    return to_keep
