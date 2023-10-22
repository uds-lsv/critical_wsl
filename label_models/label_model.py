import copy
import numpy as np
from sklearn.metrics import classification_report
from datasets import load_metric
import json
import os


class LabelModel:
    # LabelModel aggregates multiple weak labels into a single noisy label

    def __init__(self, args, logger, log_dir, random_state):
        self.args = args
        self.logger = logger
        self.log_dir = log_dir
        self.label_txt_list = None
        self.l2id = None
        self.id2l = None
        self.num_classes = None
        self.r_state = random_state

    def io_id_to_bio_id(self, a):
        bio_ids = []
        last_io = -1
        for i in a:
            if i == -100:  # subtoken id, skip
                bio_ids.append(-100)
                continue
            if i == 0:
                bio_ids.append(0)
            else:
                if i == last_io:
                    bio_ids.append(int(i * 2))  # to I
                else:
                    bio_ids.append(int(i * 2 - 1))  # to B
            last_io = i
        return bio_ids

    def ner_eval(self, predictions, labels):
        metric = load_metric("seqeval")

        # predictions = np.argmax(predictions, axis=2)
        predictions = [self.io_id_to_bio_id(p) for p in predictions]
        labels = [self.io_id_to_bio_id(l) for l in labels]

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2l[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2l[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if self.args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "overall_f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    def tc_eval(self, predictions, labels, output_dict=True):
        classification_score_dict = classification_report(labels,
                                                          np.array(predictions).flatten(),
                                                          target_names=self.label_txt_list,
                                                          digits=4,
                                                          output_dict=output_dict)
        return classification_score_dict

    def flatten(self, probs):
        L = []  # weak labels
        indexes = [0]
        for i in range(len(probs)):
            L += list(probs[i])
            indexes.append(len(probs[i]))
        indexes = np.cumsum(indexes)
        return np.array(L), indexes

    def predict_proba_ner(self, dataset, weight=None, ABSTAIN=-1):
        vote_matrices = []
        probas = []
        weak_labels = dataset.weak_labels
        n_class = dataset.num_classes

        for weak_label in weak_labels:
            L = np.array(weak_label)
            weight = np.ones_like(L)
            n, m = L.shape
            Y_p = np.zeros((n, n_class))
            Y_p_for_normalization = np.zeros((n, n_class))
            for i in range(n):
                counts = np.zeros(n_class)
                for j in range(m):
                    if L[i, j] != ABSTAIN:
                        counts[L[i, j]] += 1 * weight[i, j]

                Y_p[i, :] = counts
                if counts.sum() == 0:
                    counts += 1
                Y_p_for_normalization[i, :] = counts

            Y_p_normalized = Y_p_for_normalization / (Y_p_for_normalization.sum(axis=1, keepdims=True))

            vote_matrices.append(Y_p)
            probas.append(Y_p_normalized)

        return {'weak_labels': weak_labels, 'vote_matrices': vote_matrices, 'probas': probas}

    def predict_proba(self, dataset, weight=None, ABSTAIN=-1):
        # This function gets the probability of the noisy labels
        L = np.array(dataset.weak_labels)
        if weight is None:
            weight = np.ones_like(L)

        n_class = dataset.num_classes
        n, m = L.shape
        Y_p = np.zeros((n, n_class))
        Y_p_for_normalization = np.zeros((n, n_class))
        for i in range(n):
            counts = np.zeros(n_class)
            for j in range(m):
                if L[i, j] != ABSTAIN:
                    counts[L[i, j]] += 1 * weight[i, j]

            # the original y_p is the counts of the labels
            Y_p[i, :] = counts

            # to avoid division by zero
            if counts.sum() == 0:
                counts += 1
            Y_p_for_normalization[i, :] = counts

        Y_p_normalized = Y_p_for_normalization / (Y_p_for_normalization.sum(axis=1, keepdims=True))

        return {'weak_labels': L, 'vote_matrix': Y_p, 'proba': Y_p_normalized}

    def process_dataset(self, full_dataset, weak_labels_dict):
        train_set, val_set, test_set = full_dataset["train_set"], \
                                       full_dataset["validation_set"], \
                                       full_dataset["test_set"]
        self.l2id, self.id2l = full_dataset["l2id"], full_dataset["id2l"]
        train_weak_res, validation_weak_res, test_weak_res = weak_labels_dict["train_weak_data"], \
                                                             weak_labels_dict["validation_weak_data"], \
                                                             weak_labels_dict["test_weak_data"]

        n_labels = copy.deepcopy(train_weak_res['aggregated_labels'])
        train_set.n_labels = n_labels

        train_set.gen_bert_input()
        l_set, ul_set = train_set.get_covered_subset()

        val_set.n_labels = copy.deepcopy(validation_weak_res['aggregated_labels'])
        val_set.gen_bert_input()
        test_set.n_labels = copy.deepcopy(test_weak_res['aggregated_labels'])
        test_set.gen_bert_input()

        return {'l_set': l_set, 'ul_set': ul_set, 'validation_set': val_set, 'test_set': test_set}

    def analyze_weak_labels(self, full_dataset, weak_labels_res):
        train_set, val_set, test_set = full_dataset["train_set"], \
                                       full_dataset["validation_set"], \
                                       full_dataset["test_set"]
        self.l2id, self.id2l = full_dataset["l2id"], full_dataset["id2l"]

        train_weak_res, validation_weak_res, test_weak_res = weak_labels_res["train_weak_data"], \
                                                             weak_labels_res["validation_weak_data"], \
                                                             weak_labels_res["test_weak_data"]

        train_set.n_labels = copy.deepcopy(train_weak_res['aggregated_labels'])
        val_set.n_labels = copy.deepcopy(validation_weak_res['aggregated_labels'])
        test_set.n_labels = copy.deepcopy(test_weak_res['aggregated_labels'])

        # save train, validation and test weak labels to disk
        for d_set, d_name in zip([train_set, val_set, test_set], ['train', 'validation', 'test']):
            weak_labels = d_set.weak_labels
            labels = d_set.labels
            n_labels = d_set.n_labels
            text = [e['text'] for e in d_set.examples]
            id = d_set.ids
            rows = []
            for w, l, nl, t, i in zip(weak_labels, labels, n_labels, text, id):
                row = {'text': t, 'labels': l, 'n_labels': nl, 'id': i, 'weak_labels': w}
                rows.append(row)
            # save to json
            output_dir = './data/wrench/hf_format'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{d_name}.json")
            with open(output_path, 'w') as f:
                json.dump(rows, f)

        for weak_labels_data, d, d_name in zip([train_weak_res, validation_weak_res, test_weak_res],
                                               [train_set, val_set, test_set], ['train', 'validation', 'test']):

            print(f"{d_name} weak label statistics:")

            weak_labels_mat = weak_labels_data["weak_labels"]
            # Note that if no rule is triggered for instance i, the i-th row of weak_labels_mat contains all ones
            if self.args.task_type == "ner":
                gt_labels_ner = copy.deepcopy(np.array(d.labels))
                gt_labels = np.hstack(gt_labels_ner)
                vote_matrices = weak_labels_data["vote_matrices"]
                vote_matrix = np.vstack(vote_matrices)
                weak_labels_ner = weak_labels_data["aggregated_labels"]
                n_labels = np.hstack(weak_labels_ner)
            else:
                gt_labels = copy.deepcopy(np.array(d.labels))
                vote_matrix = (weak_labels_data["vote_matrix"]).astype(int)
                n_labels = weak_labels_data["aggregated_labels"]

            # Coverage
            # for majority voting and at-least-one voting, it should match the number of triggered examples.
            # for majority voting without tie, the coverage should be smaller.
            all_abstain_mask = (n_labels == -1)
            coverage = 1 - (np.sum(all_abstain_mask) / len(n_labels))
            print("Covered samples:", np.sum(~all_abstain_mask))
            print(f"Coverage Percentage: {coverage}")

            # Conflict Fraction
            labeled_vote_matrix = vote_matrix[~all_abstain_mask]
            conflicts = np.sum(vote_matrix != 0, axis=1) > 1
            conflicts = np.logical_and(conflicts, ~all_abstain_mask)
            conflicted_samples = np.sum(conflicts)
            conflicts_fraction = conflicted_samples / (len(labeled_vote_matrix))
            print(f"Number of Conflicted Samples: {conflicted_samples}")
            print(f"Conflict Fraction: {conflicts_fraction}")

            # Tie Fraction
            # note tie always means conflict, but not the other way around
            # compute tie candidates: they are the samples which have the same most voted label.
            # but this computation will also include not-voted samples, which are not tie samples.

            most_voted = np.max(vote_matrix, axis=1)  # the most voted label for each sample
            tie_candidates_mask = (vote_matrix == most_voted[:, None])
            tie_candidates_mask = np.sum(tie_candidates_mask, axis=1) > 1

            # the real tie samples come from the candidates, but exclude the not-voted samples.
            tie_samples_mask = np.logical_and(conflicts, tie_candidates_mask)
            tie_fraction = np.sum(tie_samples_mask) / (len(labeled_vote_matrix))
            print(f"Number of Tie Samples: {np.sum(tie_samples_mask)}")
            print(f"Tie Fraction: {tie_fraction}")

            # Accuracy, Precision, Recall, F1
            covered_n_labels = n_labels[~all_abstain_mask]
            covered_gt_labels = gt_labels[~all_abstain_mask]
            target_names = [self.id2l[i] for i in range(len(self.id2l))]
            if self.args.task_type == "ner":
                assert coverage == 1.0
                covered_cp = self.ner_eval(weak_labels_ner, gt_labels_ner)
                print(f"Coverage NER Report: {covered_cp}")
            else:
                covered_cp = self.tc_eval(covered_n_labels, covered_gt_labels, output_dict=False)
                print(f"Covered Classification Report: \n{covered_cp}")

    def get_weak_labels(self, full_dataset):
        raise NotImplementedError("implement it in sub classes")

    def aggregate_labels(self, args, logger, full_dataset):
        logger.info('Majority Vote Label Model: aggregating weak labels')
        weak_labels_dict = self.get_weak_labels(full_dataset)
        return weak_labels_dict

    def evaluate_label_model(self, args, logger, full_dataset):
        logger.info('Majority Vote Label Model: aggregating and analyzing weak labels')
        weak_labels_dict = self.get_weak_labels(full_dataset)
        self.analyze_weak_labels(full_dataset, weak_labels_dict)

    def train(self, args, logger, full_dataset):
        return self.evaluate_label_model(args, logger, full_dataset)
