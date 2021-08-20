import tensorflow.compat.v1 as tf

tf.get_logger().setLevel("ERROR")
tf.disable_v2_behavior()

from magenta.models.score2perf import score2perf

import numpy as np

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib


class UnconditionalGenerator():
    class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
        @property
        def add_eos_symbol(self):
            return True

    def __init__(self):
        problem = self.PianoPerformanceLanguageModelProblem()
        self.unconditional_encoders = problem.get_feature_encoders()

        hparams = trainer_lib.create_hparams(hparams_set='transformer_tpu')
        trainer_lib.add_problem_hparams(hparams, problem)
        hparams.num_hidden_layers = 16
        hparams.sampling_method = 'random'

        decode_hparams = decoding.decode_hparams()
        decode_hparams.alpha = 0.0
        decode_hparams.beam_size = 1

        run_config = trainer_lib.create_run_config(hparams)
        estimator = trainer_lib.create_estimator('transformer', hparams, run_config, decode_hparams=decode_hparams)

        self.targets = []
        self.decode_length = 0

        input_fn = decoding.make_input_fn_from_generator(self.input_generator())
        self.unconditional_samples = estimator.predict(input_fn, checkpoint_path='gs://magentadata/models/music_transformer/checkpoints/unconditional_model_16.ckpt')
        _ = next(self.unconditional_samples)

    def input_generator(self):
        while True:
            yield {
                'targets': np.array([self.targets], dtype=np.int32),
                'decode_length': np.array(self.decode_length, dtype=np.int32)
            }

    def decode(ids, encoder):
        ids = list(ids)
        if text_encoder.EOS_ID in ids:
            ids = ids[:ids.index(text_encoder.EOS_ID)]
        return encoder.decode(ids)

    def generate(self, decode_length=1024):
        self.targets = []
        self.decode_length = decode_length
        sample_ids = next(self.unconditional_samples)['outputs']

        ids = list(sample_ids)
        if text_encoder.EOS_ID in ids:
            ids = ids[:ids.index(text_encoder.EOS_ID)]
        midi = self.unconditional_encoders['targets'].decode(ids)

        return midi


if __name__ == "__main__":
    generator = UnconditionalGenerator()
    ns = generator.generate()
    print(ns)
