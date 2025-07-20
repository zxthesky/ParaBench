import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from transformers import AutoTokenizer
import datasets
import fire
import numpy as np
import transformers
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

# from .data import TaskPlannerDatasetBuilder
from .data import TaskPlanner
from .model import TaskPlannerModel

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainingArguments(TrainingArguments):
    plm_dir: str = field(default="microsoft/deberta-v3-large")
    data_dir: str = field(default="Spico/TaskLAMA")
    cache_dir: str = field(default=None)
    max_seq_len: int = field(default=512)


def safe_div(num, denom):
    if denom == 0:
        return 0.0
    return num / denom


def compute_metrics(ep: EvalPrediction):
    n_candidate_links = n_pred_links = tp = fp = fn = acc_num = 0
    acc_links = all_cls_links = 0
    predictions, label_ids = ep
    n_total = len(predictions)
    
    print("????????????????????")
    print("predictions.shape")
    print(predictions.shape)
    print("label_ids.shape")
    print(label_ids.shape)
    
    for logits, _labels in zip(predictions, label_ids):
        labels = _labels.copy()  # copy to avoid modifying the original labels
        n_candidate_links += labels[labels != -100].sum()
        print("logits.shape")
        print(logits.shape)
        pred = logits.argmax(0)
        print("pred.shape")
        print(pred.shape)
        all_cls_links += (labels != -100).astype(np.int64).sum()
        eq_links = (pred == labels).astype(np.int64)
        eq_links[labels == -100] = 0
        acc_links += eq_links.sum()
        pred[labels == -100] = 0
        labels[labels == -100] = 0

        if (pred == labels).all():
            acc_num += 1

        n_pred_links += pred.sum()
        _tp = pred * labels
        tp += _tp.sum()
        fp += (pred - _tp).sum()
        fn += (labels - _tp).sum()
    dadsa

    return {
        "n_pred_links": n_pred_links,
        "p": safe_div(tp, tp + fp),
        "r": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "acc": safe_div(acc_num, n_total),
        "link_acc": safe_div(acc_links, all_cls_links),
    }


def get_links(pred: np.ndarray) -> list[tuple[int, int]]:
    # pred: (seq_len, seq_len)
    return np.stack(np.where(pred == 1), axis=0).T.tolist()


def get_original_links(links, mask):
    node2id = {}
    node_id = 0
    for i, el in enumerate(mask):
        if el == 6:
            node2id[i] = node_id
            node_id += 1
    ori_links = []
    for link in links:
        ori_links.append((node2id[link[0]], node2id[link[1]]))
    ori_links.sort()
    return ori_links


def main(config_filepath="conf/train.yaml"):
    parser = HfArgumentParser(EnhancedTrainingArguments)
    args: EnhancedTrainingArguments = parser.parse_yaml_file(config_filepath)[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    set_seed(args.seed)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {args.parallel_mode.value == 'distributed'}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # data_builder = TaskPlannerDatasetBuilder(
    #     args.plm_dir,
    #     data_dir=args.data_dir,
    #     max_seq_len=args.max_seq_len,
    #     cache_dir=args.cache_dir,
    # )
    # train_dataset = data_builder.get_dataset("train")
    # eval_dataset = data_builder.get_dataset("validation")
    # test_dataset = data_builder.get_dataset("test")
    tokenizer = AutoTokenizer.from_pretrained(args.plm_dir)
    print("this  change !!!!!!!!!!!!!!!!!")
    train_dataset = TaskPlanner("/data/xzhang/task_planning/main/data/train_data/train_data_3000.json", tokenizer)
    eval_dataset = TaskPlanner("/data/xzhang/task_planning/main/data/train_data/train_data_3000.json", tokenizer)
    test_dataset = TaskPlanner("/data/xzhang/task_planning/main/data/test_data/test.json", tokenizer)

    model = TaskPlannerModel(args.plm_dir, len(tokenizer))

    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=data_builder.tokenizer,
    #     data_collator=data_builder.collate_fn,
    #     compute_metrics=compute_metrics,
    # )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=train_dataset.collate_fn,
        compute_metrics=compute_metrics,
    )

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Train Evaluate ***")
        # trainer._load_from_checkpoint("outputs/debug")
        metrics = trainer.evaluate(train_dataset)
        metrics["train_eval_samples"] = len(train_dataset)
        trainer.log_metrics("train_eval", metrics)
        trainer.save_metrics("train_eval", metrics)

        logger.info("*** Dev Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        metrics["dev_eval_samples"] = len(eval_dataset)
        trainer.log_metrics("dev_eval", metrics)
        trainer.save_metrics("dev_eval", metrics)

        logger.info("*** Test Evaluate ***")
        metrics = trainer.evaluate(test_dataset)
        metrics["test_eval_samples"] = len(test_dataset)
        trainer.log_metrics("test_eval", metrics)
        trainer.save_metrics("test_eval", metrics)

    # Predict
    if args.do_predict:
        logger.info("*** Predict ***")
        # trainer._load_from_checkpoint("outputs/debug")
        predictions, labels, metrics = trainer.predict(
            test_dataset, metric_key_prefix="test"
        )
        metrics["predict_samples"] = len(test_dataset)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_test_predictions_file = Path(args.output_dir) / "test_predictions.txt"
        output_test_predictions_file_json = Path(args.output_dir) / "test_predictions.json"


        all_datas = []
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w", encoding="utf8") as fout:
                for _pred, _label, ins in zip(predictions, labels, test_dataset):
                    pred = _pred.copy()
                    label = _label.copy()

                    pred = pred.argmax(0)
                    pred[label == -100] = 0
                    label[label == -100] = 0

                    pred_links = get_links(pred)
                    gold_links = get_links(label)
                    mask = ins["mask"]
                    ori_pred_links = get_original_links(pred_links, mask)
                    ori_gold_links = get_original_links(gold_links, mask)

                    res = {
                        "pred": ori_pred_links,
                        "gold": ori_gold_links,
                        "raw": ins,
                    }

                    all_datas.append(res)
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")

            with open(output_test_predictions_file_json, "w", encoding="utf-8") as f_json:
                json.dump(all_datas, f_json, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
