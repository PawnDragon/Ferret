import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from utils_data.default_tokens import DefaultToken
from utils_data.model_loader import (
    is_qwen3_model,
    maybe_print_qwen3_selfcheck,
    resolve_model_source,
)
from utils_data.partition_data import partition_idx_labeldir
from collections import Counter


def get_loaders(args, only_eval=False):
    """
    Return: list of train_loaders, eval_loader
    """
    model_source = resolve_model_source(args.model)
    is_qwen3 = is_qwen3_model(model_source)
    tokenizer_kwargs = {'use_fast': True}
    if is_qwen3:
        tokenizer_kwargs['trust_remote_code'] = True
    tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None and (not is_qwen3):
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_qwen3:
        maybe_print_qwen3_selfcheck(tokenizer, model_source)

    # Generation task
    if args.dataset == 'dolly':
        from utils_data.llm_dataset import LLMDataset, LLMDataCollator
        if args.eval_metric == 'loss':
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=False)
        else:
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=True)

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        # only use a subset of raw dataset
        raw_datasets, _ = torch.utils.data.dataset.random_split(raw_datasets, [int(len(raw_datasets) * args.dataset_subsample), len(raw_datasets) - int(len(raw_datasets) * args.dataset_subsample)])
        y_all = np.array([item['categories'] for item in raw_datasets])
        index_eval = np.where(y_all == args.zerotask)[0]
        # delete the indices of eval samples from the all set
        index_train = np.delete(np.arange(len(y_all)), index_eval)
        raw_datasets = np.array(raw_datasets)
        train_set = raw_datasets[index_train]
        eval_set = raw_datasets[index_eval]
        y_train = np.array([item['categories'] for item in train_set])
        counter = Counter(y_train)
        noniid = args.iid
        if 'dir' in noniid:
            split_dic = partition_idx_labeldir(y_train, n_parties=args.num_clients, alpha=float(noniid[3:]), num_classes=len(counter))
            split_trainsets = []
            for _, sample_indices in split_dic.items():
                split_trainsets.append(Subset(train_set, indices=sample_indices))
        else:
            n_parts = [int(len(train_set) / args.num_clients) for _ in range(args.num_clients - 1)]
            n_parts.append(len(train_set) - sum(n_parts))
            split_trainsets = torch.utils.data.dataset.random_split(train_set, n_parts)

        list_train_loader = [
            DataLoader(
                subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
            ) for subset in split_trainsets
        ]
        eval_loader = DataLoader(
            eval_set, batch_size=args.batch_size, collate_fn=data_collator
        )
        
    elif args.dataset == 'gsm8k':
        from utils_data.gsm8k_loader import load_gsm8k_local_splits
        from utils_data.gsm8k_metrics import gsm8k_partition_bucket_from_gold
        from utils_data.natural_instruction_loader import LLMDataset, LLMDataCollator

        train_examples, test_examples = load_gsm8k_local_splits(getattr(args, 'gsm8k_root', './data/gsm8k'))
        train_data = [(item['instruction'], item['input'], item['output']) for item in train_examples]
        eval_data = [(item['instruction'], item['input'], item['output']) for item in test_examples]
        train_labels = np.array([gsm8k_partition_bucket_from_gold(item['output']) for item in train_examples], dtype=np.int64)

        data_collator = LLMDataCollator(tokenizer=tokenizer)
        list_train_loader = []
        if not only_eval:
            train_dataset = LLMDataset(
                train_data,
                tokenizer=tokenizer,
                use_prompts=args.use_prompts,
                generation=False,
            )

            noniid = args.iid
            if 'dir' in noniid:
                split_dic = partition_idx_labeldir(
                    train_labels,
                    n_parties=args.num_clients,
                    alpha=float(noniid[3:]),
                    num_classes=10,
                )
                split_trainsets = []
                for _, sample_indices in split_dic.items():
                    split_trainsets.append(Subset(train_dataset, indices=sample_indices))
            else:
                n_parts = [int(len(train_dataset) / args.num_clients) for _ in range(args.num_clients - 1)]
                n_parts.append(len(train_dataset) - sum(n_parts))
                split_trainsets = torch.utils.data.dataset.random_split(train_dataset, n_parts)

            list_train_loader = [
                DataLoader(
                    subset,
                    shuffle=True,
                    batch_size=args.batch_size,
                    collate_fn=data_collator,
                )
                for subset in split_trainsets
            ]

        eval_generation = bool(args.eval_metric != 'loss')
        eval_dataset = LLMDataset(
            eval_data,
            tokenizer=tokenizer,
            use_prompts=args.use_prompts,
            generation=eval_generation,
        )
        eval_loader = DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=data_collator,
        )

    elif args.dataset == 'math':
        from utils_data.math_loader import load_math_local_splits
        from utils_data.natural_instruction_loader import LLMDataset, LLMDataCollator

        train_examples, dev_examples, test_examples = load_math_local_splits(
            dataset_path=getattr(args, 'dataset_path', './data/math'),
            seed=int(getattr(args, 'seed', 42)),
            dev_size=500,
        )
        if (not only_eval) and len(dev_examples) == 0:
            raise RuntimeError(
                '[math] dev split is empty for round evaluation; please provide a larger train split'
            )

        train_data = [(item['instruction'], item['input'], item['output']) for item in train_examples]
        eval_examples = test_examples if only_eval else dev_examples
        eval_data = [(item['instruction'], item['input'], item['output']) for item in eval_examples]
        eval_metadata = [
            {
                'meta_ref_solution': item['output'],
                'meta_final_answer': item['final_answer'],
                'meta_subject': item['subject'],
                'meta_level': int(item['level']),
            }
            for item in eval_examples
        ]

        label_keys = [(str(item['subject']), int(item['level'])) for item in train_examples]
        unique_keys = sorted(set(label_keys))
        key_to_label = {key: idx for idx, key in enumerate(unique_keys)}
        train_labels = np.array([key_to_label[key] for key in label_keys], dtype=np.int64)
        num_classes = max(len(unique_keys), 1)

        data_collator = LLMDataCollator(tokenizer=tokenizer)
        list_train_loader = []
        if not only_eval:
            train_dataset = LLMDataset(
                train_data,
                tokenizer=tokenizer,
                use_prompts=args.use_prompts,
                generation=False,
            )
            noniid = args.iid
            if 'dir' in noniid:
                split_dic = partition_idx_labeldir(
                    train_labels,
                    n_parties=args.num_clients,
                    alpha=float(noniid[3:]),
                    num_classes=num_classes,
                )
                split_trainsets = []
                for _, sample_indices in split_dic.items():
                    split_trainsets.append(Subset(train_dataset, indices=sample_indices))
            else:
                n_parts = [int(len(train_dataset) / args.num_clients) for _ in range(args.num_clients - 1)]
                n_parts.append(len(train_dataset) - sum(n_parts))
                split_trainsets = torch.utils.data.dataset.random_split(train_dataset, n_parts)

            list_train_loader = [
                DataLoader(
                    subset,
                    shuffle=True,
                    batch_size=args.batch_size,
                    collate_fn=data_collator,
                )
                for subset in split_trainsets
            ]

        eval_generation = bool(args.eval_metric != 'loss')
        eval_dataset = LLMDataset(
            eval_data,
            tokenizer=tokenizer,
            use_prompts=args.use_prompts,
            generation=eval_generation,
            metadata_list=(eval_metadata if eval_generation else None),
        )
        eval_loader = DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=data_collator,
        )

    elif args.dataset in ['instruct']:
        from utils_data.natural_instruction_loader import get_instruction_dataset
        list_train_loader, eval_loader = get_instruction_dataset(args, tokenizer, only_eval=only_eval)
    else:
        raise AttributeError(f'dataset {args.dataset} not implemented')
    return list_train_loader, eval_loader, tokenizer
