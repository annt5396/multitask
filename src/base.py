from transformers import Trainer, is_torch_tpu_available
import datasets
from typing import Optional, List, Dict, Tuple, Callable, Any, Union
from transformers.trainer_utils import (
    PredictionOutput,
    EvalPrediction,
    EvalLoopOutput,
    denumpify_detensorize,
    speed_metrics,
)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class BaseReader(Trainer):
    name: str = None

    def __init__(
        self, *args,
        data_args = {},
        eval_examples: datasets.Dataset = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_args = data_args
        self.eval_examples = eval_examples
    
    def post_process_function(
        self,
        output: EvalLoopOutput,
    ) -> Union[Any, EvalPrediction]:
        return output
    
    def evaluate(
        self,
        eval_dataset: Optional[datasets.Dataset] = None,
        eval_examples: Optional[datasets.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(output, eval_examples, eval_dataset)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
    
    def predict(
        self,
        test_dataset: datasets.Dataset,
        test_examples: datasets.Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        mode: bool = "predict",
    ) -> PredictionOutput:
        test_dataloader = self.get_test_dataloader(test_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                test_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics
        
        predictions = self.post_process_function(output, test_examples, test_dataset, mode=mode)

        return predictions