import os
import time

import torch
import torchvision
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import eval_single_dataset, get_n_shots
from src.modeling import get_model
from src.utils import cosine_lr, extract_names, GaussianBlur, InfiniteRandomSubsetSampler, convert_to_rgb
import src.utils as utils
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)
    model = get_model(args)

    dataset = get_dataset(
        train_dataset,
        model.encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    val_loader = dataset.test_loader
    
    dataset = get_dataset(
        train_dataset.replace('Val', ''),
        model.encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    test_loader = dataset.test_loader

    #TIP augs:0.5 CLIP-LoRA augs: 0.08
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + model.encoder.val_preprocess.transforms[-3:])
        
    if args.base_augs:        
        preprocess_fn = model.encoder.train_preprocess
        
    if args.strong_augs:
        #NoLA Augs
        preprocess_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            convert_to_rgb,
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            torchvision.transforms.RandomHorizontalFlip()] + model.encoder.val_preprocess.transforms[-2:])
               
        
    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
            
    if not isinstance(args.data_ratio, float):
        to_keep = get_n_shots(dataset.train_dataset, args.data_ratio, model.classifier.out_features, args)
        print(f"Got {len(to_keep):,} trusted samples")
        r = len(to_keep) / args.batch_size
        if r < 10: #Oversampling epochs for very low shots
            over_sampling = 10/r
            over_sampling = int(over_sampling) + 1
            print(f"Oversampling {over_sampling} times")
            to_keep = torch.cat([to_keep] * over_sampling)
            
        sampler = InfiniteRandomSubsetSampler(to_keep, len(to_keep)*args.epochs)
        data_loader = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    else:
        subset = torch.randperm(len(dataset.train_dataset))[:int(args.data_ratio * len(dataset.train_dataset))]
        subset.sort()
        sampler = InfiniteRandomSubsetSampler(subset, len(subset)*args.epochs)
        data_loader = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    loss_fn = torch.nn.CrossEntropyLoss()

    params_encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    num_param = sum([p.numel() for p in params_encoder])
    params_classifier = [p for p in model.classifier.parameters() if p.requires_grad]
    
    if len(params_classifier) > 0:#DinoV2 or CLIPLP
        num_paramc = sum([p.numel() for p in params_classifier])
        
        print(f"Training {num_param:,} encoder parameters and {num_paramc:,} classifier. Total {num_param + num_paramc:,}")
        if args.lp_clip:
            optimizer = torch.optim.AdamW([{'params':params_encoder},
                                           {'params':params_classifier, 'lr':0.02}], lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = torch.optim.AdamW([{'params':params_encoder},
                                           {'params':params_classifier, 'lr':1e-3}], lr=args.lr, weight_decay=args.wd)
    else:
        print(f"Training {num_param:,} parameters")    
        optimizer = torch.optim.AdamW(params_encoder, lr=args.lr, weight_decay=args.wd)        

    scheduler = cosine_lr(optimizer, len(data_loader) // args.num_grad_accumulation)
    
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.num_grad_accumulation)

    
    model, optimizer, scheduler, data_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, scheduler, data_loader, val_loader, test_loader)
    args.accelerator = accelerator
    
    train_time = time.time()
    max_mem, loss = 0, torch.tensor([0.])
    
    model.train()
    tbar = tqdm(data_loader, disable=args.no_tqdm, leave=False)
    tbar.set_description(f"Iteration: 0/{len(data_loader)}, Loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.4f}, memory allocated {torch.cuda.memory_reserved()/1000000000:.2f}G")

    #Training with no early stopping
    for i, batch in enumerate(tbar):
        with accelerator.accumulate(model):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"]
            inputs = inputs.half().cuda()                
            labels = batch["labels"]

            loss = model(inputs, labels)
            
            accelerator.backward(loss)                
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if i % 10 == 0:
                tbar.set_description(f"Iteration: {i}/{len(data_loader)}, Loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.4f}, memory allocated {torch.cuda.memory_reserved()/1000000000:.2f}G")
            
    train_time = time.time() - train_time
    #Test
    inf_time = time.time()    
    metrics = eval_single_dataset(model, test_loader, args)
    inf_time = time.time() - inf_time
    
    string = [f"{train_dataset.replace('Val','')} train time {train_time:.1f}s and test time {inf_time:.1f}s. Num parameters {num_param}, Max memory usage {max_mem:.2f}G.\n",f"{train_dataset.replace('Val','')}\n", f"Best accuracy {metrics['top1']*100.:.2f}\n"]
    
    print("".join(string))

    if not args.no_log:
        with open(os.path.join(args.fname, str(args.data_ratio), str(args.seed), "results.txt"), 'a') as f:
            f.writelines(string)

    if args.save_weights:#Optional model save
        #torch.save(model.state_dict(), os.path.join(args.fname, str(args.data_ratio), str(args.seed), 'weights.pth'))
        #Saves the whole model
        del model.templates #Contains lambda functions that can't be pickled
        torch.save(model, os.path.join(args.fname, str(args.data_ratio), str(args.seed), f'model_{args.train_dataset}.pth'))
    
    return metrics['top1']


if __name__ == "__main__":
    args = parse_arguments()
    
    #Same training epochs as in aTLAS (Zhang et al., NeurIPS 2024)
        
    if not isinstance(args.data_ratio, float):
        epochs = {
            "Cars": args.epochs,
            "DTD": args.epochs,
            "EuroSAT": args.epochs,
            "GTSRB": args.epochs,
            "MNIST": args.epochs,
            "RESISC45": args.epochs,
            "SUN397": args.epochs,
            "SVHN": args.epochs,
            "CIFAR10": args.epochs,
            "CIFAR100": args.epochs,        
            "STL10": args.epochs,
            "Food101": args.epochs,
            "Caltech256": args.epochs,
            "FGVCAircraft": args.epochs,
            "Flowers102": args.epochs,
            "OxfordIIITPet": args.epochs,
            "CUB200": args.epochs,
            "PascalVOC": args.epochs,
            "Country211": args.epochs,
            "Caltech101": args.epochs,
            "UCF101": args.epochs,
            "ImageNet": args.epochs,
        }
    else:
        epochs = {
            "Cars": 35,
            "DTD": 76,
            "EuroSAT": 13,
            "GTSRB": 11,
            "MNIST": 5,
            "RESISC45": 15,
            "SUN397": 14,
            "SVHN": 4,
            "CIFAR10": 5,
            "CIFAR100": 6,
            "STL10": 4,
            "Food101": 15,
            "Caltech256": 8,
            "FGVCAircraft": 60,
            "Flowers102": 40,
            "OxfordIIITPet": 5,
            "CUB200": 20,
            "PascalVOC": 10,
            "Country211": 15,
            "Caltech101":10,
            "UCF101": 20,
            "ImageNet": 10,           
        }

    if args.datasets is not None: #Train on specific datasets
        epochs =  {k:v for k,v in epochs.items() if k in args.datasets}

    if args.param_type in ['vera', 'lora']:
        args.num_basis = 1
    elif args.param_type in ['nola']:
        args.num_basis = 1024
    elif args.param_type in ['randlora']:
        args.num_basis = 1000 #Later automatically adjusted for each layer
    else:
        raise NotImplementedError

    if args.fname is not None:
        filename = os.path.join(args.fname, str(args.data_ratio), str(args.seed), "results.txt")
        if os.path.isfile(filename):
            if not args.merge:
                os.remove(filename)
            else:
                trained_on = extract_names(filename)
                epochs =  {k:v for k,v in epochs.items() if k not in trained_on}
        else:
            os.makedirs('/'.join(filename.split('/'))[:-1], exist_ok=True)
            
    args.save = f"checkpoints/{args.model}"
    for dataset in epochs:            
        args.epochs = epochs[dataset]        
        args.train_dataset = dataset

        # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
        args.num_grad_accumulation = 128 // args.batch_size
        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        
        seed = int(torch.exp(torch.tensor(1))*3.1415*1000)
        set_seed(seed)
        
        finetune(args)
        
    if not args.no_log:
        with open(os.path.join(args.fname, str(args.data_ratio), str(args.seed), "results.txt"), 'a') as f:
            f.writelines([f"{args}"+"\n"])
