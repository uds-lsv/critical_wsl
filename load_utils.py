from cwsl_dataset import TextBertDataset, NERBertDataset, REBertDataset
from transformers import AutoTokenizer
from pathlib import Path
import json

def load_json(input_json_path):
    # read file
    with open(input_json_path, 'r') as myfile:
        data = myfile.read()

    # parse file
    obj = json.loads(data)

    return obj


def load_tokenizer(args):
    if "roberta" in args.model_name and args.task_type == 'ner':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    return tokenizer

def get_data_by_tag(args, logger, tokenizer, id2l,  r_state, num_classes, tag='train'):
    file_path = Path(args.data_root) / args.dataset / f"{tag}.json"
    logger.info(f'loading data from {file_path}')
    input_data = json.load(open(file_path, 'r'))
    if args.task_type in ['text_cls', 'text_cls_f1']:
        bert_dataset = TextBertDataset(args, input_data, tokenizer, id2l)
    elif args.task_type == 'ner':
        bert_dataset = NERBertDataset(args, input_data, tokenizer, id2l)
    elif args.task_type == 're':
        bert_dataset = REBertDataset(args, input_data, tokenizer, id2l)
    else:
        raise ValueError("[load_utils]: unknown task_type")

    return bert_dataset



def prepare_data(args, logger, r_state):

    if args.task_type in ['text_cls', 'text_cls_f1', 're']:
        tokenizer = load_tokenizer(args)
        label_path = Path(args.data_root) / args.dataset / f'label.json'
        id2l = {int(k): v for k, v in json.load(open(label_path, 'r')).items()}
        l2id = {v: k for k, v in id2l.items()}
        num_classes = len(id2l.keys())
    elif args.task_type == 'ner':
        tokenizer = load_tokenizer(args)
        meta_data_path = Path(args.data_root) / args.dataset / f'meta.json'
        meta_data = load_json(meta_data_path)
        num_classes = meta_data["num_labels"]
        entity_types = meta_data["entity_types"]
        extended_entity_types = ['O']
        for et in entity_types:
            extended_entity_types.append(f'B-{et}')
            extended_entity_types.append(f'I-{et}')
        # extended_entity_types.append('O')
        assert len(extended_entity_types) == num_classes
        id2l = {idx: label for idx, label in enumerate(extended_entity_types)}
        l2id = {v: k for k, v in id2l.items()}

        for i in range(len(id2l)):
            if i == 0:
                assert id2l[i] == "O", "label O should get the index zero"
            elif i % 2 == 0:
                assert id2l[i].startswith("I-"), "we assume labels starting with I to have even indices"
            else:
                assert id2l[i].startswith("B-"), "we assume labels starting with B to have odd indices"

    else:
        raise ValueError("[load_utils]: unknown task_type")

    train_set = get_data_by_tag(args, logger, tokenizer, id2l, r_state, num_classes, tag='train')
    validation_set = get_data_by_tag(args, logger, tokenizer, id2l, r_state, num_classes, tag='valid')
    test_set = get_data_by_tag(args, logger, tokenizer, id2l, r_state, num_classes, tag='test')

    full_dataset = {"train_set": train_set, "validation_set": validation_set, "test_set": test_set,
                    "l2id": l2id, "id2l": id2l}

    return full_dataset