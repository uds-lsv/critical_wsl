import numpy as np
from snorkel.utils import probs_to_preds
from label_models.label_model import LabelModel


class MajorityVotingLabelModel(LabelModel):
    def __init__(self, args, logger, log_dir, random_state):
        super(MajorityVotingLabelModel, self).__init__(args, logger, log_dir, random_state)

    def predict(self, dataset, **kwargs) -> np.ndarray:
        """Method for predicting on given dataset.

        Parameters
        ----------
        """

        if kwargs['task_type'] == 'ner':  # ner task
            res = self.predict_proba_ner(dataset)
            # actually, flatten is not that necessary
            # but please do it, since the tie function in probs_to_preds needs the correct length as random seed
            # flatten used for reproducibility
            probas_flatten, indexes = self.flatten(res["probas"])
            majority_preds_flatten = probs_to_preds(probs=np.array(probas_flatten))
            majority_preds = [list(majority_preds_flatten[start:end]) for (start, end) in
                              zip(indexes[:-1], indexes[1:])]
            aggregated_labels = majority_preds
            aggregated_labels_flatten = majority_preds_flatten
            res['aggregated_labels'] = aggregated_labels
            res['aggregated_labels_flatten'] = aggregated_labels_flatten
            return res
        elif kwargs['task_type'] in ['text_cls', 'text_cls_f1', 're']:
            # We first vote, then correcting the noisy labels.
            # Labels that needs to be corrected: if all rules abstain, we should assign -1.
            res = self.predict_proba(dataset)
            aggregated_labels = probs_to_preds(probs=res['proba'])

            # a) assign all abstained labels to -1
            vote_matrix = (res["vote_matrix"]).astype(int)
            voting_sum = np.sum(vote_matrix, axis=1)
            all_abstain_mask = (voting_sum == 0)  # the samples which all rules abstain
            aggregated_labels[all_abstain_mask] = -1
            res['aggregated_labels'] = aggregated_labels

            return res
        else:
            raise ValueError("unknown task type")

    def get_weak_labels(self, full_dataset):
        train_set, val_set, test_set = full_dataset["train_set"], \
                                       full_dataset["validation_set"], \
                                       full_dataset["test_set"]
        self.l2id, self.id2l = full_dataset["l2id"], full_dataset["id2l"]

        if self.args.task_type == 'ner':
            num_classes = len(self.l2id.keys())
            assert num_classes % 2 != 0, "number of BIO classes should always be odd"
            self.num_classes = int((num_classes + 1) / 2)
        elif self.args.task_type in ['text_cls', 'text_cls_f1', 're']:
            self.num_classes = len(self.l2id.keys())
        else:
            raise ValueError("[Trainer]: Unknown task_type")

        self.logger.info("computing majority vote")
        train_mv_res = self.predict(train_set, **{"task_type": self.args.task_type})
        validation_mv_res = self.predict(val_set, **{"task_type": self.args.task_type})
        test_mv_res = self.predict(test_set, **{"task_type": self.args.task_type})

        return {"train_weak_data": train_mv_res,
                "validation_weak_data": validation_mv_res,
                "test_weak_data": test_mv_res}
