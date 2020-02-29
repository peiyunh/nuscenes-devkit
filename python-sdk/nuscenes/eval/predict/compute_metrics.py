# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
""" Script for computing metrics for a submission to the nuscenes prediction challenge. """
import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.predict.config import PredictionConfig
from nuscenes.eval.predict.data_classes import Prediction
from nuscenes.predict import PredictHelper


def compute_metrics(predictions: List[Dict[str, Any]],
                    helper: PredictHelper, config: PredictionConfig) -> Dict[str, Any]:
    """
    Computes metrics from a set of predictions.
    :param predictions: Unserialized predictions in json file.
    :param helper: Instance of PredictHelper that wraps the nuScenes test set.
    :param config: Config file.
    """
    # TODO: Add check that n_preds is same size as test set once that is finalized
    n_preds = len(predictions)

    containers = {metric.name: np.zeros((n_preds, metric.shape)) for metric in config.metrics}

    for i, prediction_str in enumerate(predictions):

        prediction = Prediction.deserialize(prediction_str)
        ground_truth = helper.get_future_for_agent(prediction.instance, prediction.sample,
                                                   config.seconds, in_agent_frame=True)
        if np.all(np.sqrt(np.sum(ground_truth**2, axis=1)) < 1):
            import ipdb; ipdb.set_trace()
        for metric in config.metrics:
            containers[metric.name][i] = metric(ground_truth, prediction)

    aggregations: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for metric in config.metrics:
        for agg in metric.aggregators:
            aggregations[metric.name][agg.name] = agg(containers[metric.name])

    return aggregations


def main(version: str, data_root: str, submission_path: str, submission_name: str,
         config_name: str = 'predict_2020_icra') -> None:
    """
    Computes metrics for a submission stored in submission_path with a given submission_name with the metrics
    specified by the config_name.
    :param version: NuScenes dataset version.
    :param data_root: Directory storing NuScenes data.
    :param submission_path: Directory storing submission.
    :param submission_name: Name of json file in submission_path directory that stores
        predictions.
    :param config_name: Name of config file.
    """

    predictions = json.load(open(os.path.join(submission_path, f"{submission_name}_inference.json"), "r"))
    config = config_factory(config_name)
    nusc = NuScenes(version=version, dataroot=data_root)
    helper = PredictHelper(nusc)

    results = compute_metrics(predictions, helper, config)
    json.dump(results, open(os.path.join(submission_path, f"{submission_name}_metrics.json"), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate.')
    parser.add_argument('--version', help='NuScenes version number.')
    parser.add_argument('--data_root', help='Root directory for NuScenes json files.')
    parser.add_argument('--output_dir', help='Directory to store output file.')
    parser.add_argument('--submission_name', help='Name of the submission to use for the results file.')
    parser.add_argument('--config_name', help='Name of the config file to use', default='predict_2020_icra')
    
    args = parser.parse_args()
    main(args.version, args.data_root, args.output_dir, args.submission_name, args.config_name)