import os.path
from shutil import copy2
import data.read_datasets
import data.data_augmentation
import data.adapet_preperation
from CySecTrainer import CySecTrainer
from ADAPETCySecTrainer import ADAPETCySecTrainer
from TemplatedCySecTrainer import TemplatedCySecTrainer
import argparse, pickle, random, torch, json, os, glob
import datetime as dt
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    print(f'{dt.datetime.now()} Starting main')
    parser = argparse.ArgumentParser()
    parser.add_argument("-rp", "--root_path", type=str, default=os.getcwd(), help="path to root directory, if needed")
    parser.add_argument("-f", "--fewshot", required=True, help="fewshot dataset or full dataset")
    parser.add_argument("-a", "--augmentation", required=True, help="additional augmented dataset or not")
    parser.add_argument("-t", "--tune", required=True, help="model finetuning or not")
    parser.add_argument("-m", "--mode", required=True, help="chose from 0-2 which training method should be used."+ \
    "0: Normal training with CySecTrainer, 1: ADAPET trainig with ADAPETCySec, 2: TemplatedCySecTrainer")
    parser.add_argument("-c", "--cached_files", default=True)
    parser.add_argument("-jn", "--job_name", required=True)
    parser.add_argument("-ji", "--jobid", required=False)
    parser.add_argument("-ti", "--taskid", required=False)
    parser.add_argument("-b", "--bert", required=True,\
    help="path to the model or name of the model (must be available on hugginface.hub)")
    parser.add_argument("-ds", "--dataset", default="CySec/GeneralCySec", help="SpecializedCySec or GeneralCySec dataset")
    parser.add_argument("-mtl", "--max_text_length", default=256)
    parser.add_argument("-bs", "--batch_size", default=1)
    parser.add_argument("-ebs", "--eval_batch_size", default=1)
    parser.add_argument("-nb", "--num_batches", default=1000)
    parser.add_argument("-mbl", "--max_num_lbl", default=10)
    parser.add_argument("-mblt", "--max_num_lbl_tok", default=1)
    parser.add_argument("-ee", "--eval_every", default=250)
    parser.add_argument("-wr", "--warmup_ratio", default=0.06)
    parser.add_argument("-ma", "--mask_alpha", default=0.105)
    parser.add_argument("-gaf", "--grad_accumulation_factor",default=16)
    #parser.add_argument("s", "--seed", default=109)
    parser.add_argument("-lr", "--learning_rate", default=1e-5)
    parser.add_argument("-wd", "--weight_decay", default=1e-2)
    parser.add_argument("-p", "--pattern", default=1)
    parser.add_argument("-et", "--eval_train", default=True)
    parser.add_argument("-e", "--epochs", default=5)
    parser.add_argument("-ws", "--warmup_steps", default=200)
    parser.add_argument("-ed", "--experiment_dir", required=True, \
    help="Path to the directory where the experiment should be saved (only during ADAPET training)")
    args = parser.parse_args()
    print(f'{dt.datetime.now()} Arguments parsed')
    # Changing the root path
    if args.root_path :
        os.chdir(args.root_path)
    os.environ['ADAPET_ROOT'] = os.getcwd()+"/src/ADAPET"

    # Setting a random seed or jobid + taskid instead of default 42
    if(args.jobid and args.taskid):
        seed = int(args.jobid) + int(args.taskid)
    else:
        seed = random.randint(0,100000)
            
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Creating a parameter dict and saving the parameters in a json for ADAPET
    # Not every parameter is used in every training
    parameters = {
        "root_path":str(args.root_path),
        "pretrained_weight":str(args.bert),
        "dataset":str(args.dataset),
        "max_text_length": int(args.max_text_length),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_batches": int(args.num_batches),
        "max_num_lbl_tok": int(args.max_num_lbl_tok),
        "eval_every": int(args.eval_every),
        "warmup_ratio": float(args.warmup_ratio),
        "mask_alpha": float(args.mask_alpha),
        "grad_accumulation_factor": int(args.grad_accumulation_factor),
        "seed": seed,
        "lr": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "pattern": int(args.pattern),
        "eval_train": int(args.eval_train),
        "epochs":int(args.epochs),
        "warmup_steps": int(args.warmup_steps),
        "exp_name":str(args.job_name),
        "exp_dir":str(args.experiment_dir)
    }

    config_path="./Configs/"+parameters["exp_name"]+".json"
    with open(config_path, "w") as f:
        json.dump(parameters,f)
        
    print(f"{dt.datetime.now()} Parameters created and JSON saved")
    
    print(f'{dt.datetime.now()} Getting Dataset')
    (X_train_full, y_train_full), (X_train_few, y_train_few), (X_dev_full, y_dev_full), (X_dev_few, y_dev_few), (X_test, y_test) = data.read_datasets.read_specialized_cysec_data()
    
    if str2bool(args.fewshot):
        X_train = X_train_few
        y_train = y_train_few
        X_dev = X_dev_few
        y_dev = y_dev_few
    else:
        X_train = X_train_full
        y_train = y_train_full
        X_dev = X_dev_full
        y_dev = y_dev_full
        
    
    augmented_data_pos = []
    augmented_data_neg = []
    if str2bool(args.augmentation):
        files_found = os.path.isfile("./data/augmented/SpecializedCySec/train_pos.pkl") and os.path.isfile("./data/augmented/SpecializedCySec/train_neg.pkl")
        
        if not str2bool(args.cached_files) or not files_found:
            print("Augmenting the data is not implemented! Pull the Master and try using the already augmented datasets.")
            #augmented_data_pos = data.data_augmentation.augment_data(X_train, y_train, class_to_be_augmented=1)
            #augmented_data_pos = augmented_data_pos.apply_filtering(X_train, y_train, reference_class:1, augmented_data_pos)
            #data.data_augmentation.save_augmented_data(augmented_data_pos, 1) 
            
            #augmented_data_neg = data.data_augmentation.augment_data(X_train, y_train, class_to_be_augmented=0)
            #augmented_data_neg = augmented_data_neg.apply_filtering(
            #                                    X_train, y_train, reference_class:0, augmented_data_neg, close_instances=True)
            #data.data_augmentation.save_augmented_data(augmented_data_neg, 0)
        
        with open ('./data/augmented/SpecializedCySec/train_pos.pkl', 'rb') as fp:
            X_train_additional_pos = list(pickle.load(fp))
        with open ('./data/augmented/SpecializedCySec/train_neg.pkl', 'rb') as fp:
            X_train_additional_neg = list(pickle.load(fp))

        X_train += X_train_additional_pos + X_train_additional_neg
        y_train += [1] * len(X_train_additional_pos) + [0] * len(X_train_additional_neg)
            
    
    print(f'{dt.datetime.now()} {parameters["pretrained_weight"]} is chosen')
        
    model = None
    model_dir = None
    if args.tune == "ClassifierHead":
        # finetune bert_model with classifier head on CySecAlert dataset

        print(f"{dt.datetime.now()} Starting Stage 1")
        
        dataset, labels = data.read_datasets.read_general_cysec_data()
        (X_train_finetuning, y_train_finetuning), (X_dev_finetuning, y_dev_finetuning), (X_test_finetuning, y_test_finetuning) = \
            data.read_datasets.split_for_normal_shot(dataset, labels, build_test_set=True)
            
        finetuned_trainer = CySecTrainer(parameters["pretrained_weight"])
        finetuned_trainer.train(X_train_finetuning, y_train_finetuning, X_dev_finetuning, y_dev_finetuning, X_test_finetuning,
                                y_test_finetuning, parameters)
        model_dir = finetuned_trainer.save_model()
        model = finetuned_trainer.model
        finetuned = True
        run.finish()
        
    elif args.tune == "ADAPET":
        print(f"{dt.datetime.now()} Starting Stage 1")
        # finetune bert_model with ADAPET on CySecAlert dataset
            
        dataset, labels = data.read_datasets.read_general_cysec_data()
        (X_train_finetuning, y_train_finetuning), (X_dev_finetuning, y_dev_finetuning), (X_test_finetuning, y_test_finetuning) = \
            data.read_datasets.split_for_normal_shot(dataset, labels, build_test_set=True)
        
        data.adapet_preperation.save_datasets_for_adapet("GeneralCySec",
            X_train_finetuning, y_train_finetuning, [], [], X_dev_finetuning, y_dev_finetuning, X_test_finetuning, y_test_finetuning)
        
        return_model = args.mode != "1"
        adapet_trainer = ADAPETCySecTrainer(config_path,None, False, None)
        model = adapet_trainer.train(return_model)
        adapet_trainer.evaluate()
        model_dir = adapet_trainer.save_model()
        finetuned = True
        run.finish()
        import time
        time.sleep(100)
    
    else:
        finetuned = False
    
    if args.tune == "ADAPET" or args.tune == "ClassifierHead":
        parameters["dataset"]="CySec/SpecializedCySec"
        with open(config_path, "w") as f:
            json.dump(parameters,f)

    if args.mode == "0":
        # Normal training

        normal_trainer = CySecTrainer(parameters["pretrained_weight"], model)
        normal_trainer.train(X_train, y_train, X_dev, y_dev, X_test, y_test, parameters)
        print(normal_trainer.evaluate())        
        normal_trainer.save_model()
        
    elif args.mode == "1":
        # ADAPET training
        # setting model to None as the model will be loaded from disk
        model = None

    
        #from git import Repo
        #Repo.clone_from("https://github.com/rrmenon10/ADAPET", "ADAPET")
        
        data.adapet_preperation.save_datasets_for_adapet("SpecializedCySec",
            X_train, y_train, augmented_data_pos, augmented_data_neg, X_dev, y_dev, X_test, y_test)
        
        adapet_trainer = ADAPETCySecTrainer(config_path,None, finetuned, model_dir)
        adapet_trainer.train(return_model=False)
        adapet_trainer.evaluate()
        adapet_trainer.save_model()
        
    elif args.mode == "2":
        # Based on "Pre-trained Language Models Better Few-shot Learners"
        templated_trainer = TemplatedCySecTrainer(parameters["pretrained_weight"], model)
        templated_trainer.train(X_train, y_train, X_dev, y_dev, X_test, y_test, epochs=3, bs=5, warmup_steps=30000, weight_decay=0.01)
        print(templated_trainer.evaluate())
        templated_trainer.save_model()

    print(f"{dt.datetime.now()} End of training and evaluation")


if __name__ == "__main__":
   main()