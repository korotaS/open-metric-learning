import sys
sys.path.append('/home/korotas/projects/open-metric-learning/')

import cProfile
import pstats

import hydra
from omegaconf import DictConfig

from oml.const import HYDRA_BEHAVIOUR
from oml.lightning.pipelines.train_postprocessor import (
    postprocessor_training_pipeline,  # type: ignore
)


def profile_it(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            ret_val = func(*args, **kwargs)  # function to be profiled
        except Exception as e:
            print(e)
            ret_val = None
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(100)
        return ret_val
    return wrapper


@hydra.main(config_path=".", config_name="postprocessor_train.yaml", version_base=HYDRA_BEHAVIOUR)
def main_hydra(cfg: DictConfig) -> None:
    postprocessor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra = profile_it(main_hydra)
    main_hydra()
