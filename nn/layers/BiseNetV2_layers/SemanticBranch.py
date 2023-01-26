import tensorflow as tf
from StemBlock import StemBlock
from GatherAndExpandBlock import GatherAndExpandBlock
from ContextEmbeddingBlock import ContextEmbeddingBlock
class SemanticBranch(tf.keras.Model):
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.stem = StemBlock()
        self.stage_1 = tf.keras.Sequential([
            GatherAndExpandBlock(in_channels=16, out_channels=32, expansion_factor=6, strides=2),
            GatherAndExpandBlock(in_channels=32, out_channels=32, expansion_factor=6, strides=1),
        ])
        self.stage_2 = tf.keras.Sequential([
            GatherAndExpandBlock(in_channels=32, out_channels=64, expansion_factor=6, strides=2),
            GatherAndExpandBlock(in_channels=64, out_channels=64, expansion_factor=6, strides=1),
        ])
        self.stage_3 = tf.keras.Sequential([
            GatherAndExpandBlock(in_channels=64, out_channels=128, expansion_factor=6, strides=2),
            GatherAndExpandBlock(in_channels=128, out_channels=128, expansion_factor=6, strides=1),
            GatherAndExpandBlock(in_channels=128, out_channels=128, expansion_factor=6, strides=1),
            GatherAndExpandBlock(in_channels=128, out_channels=128, expansion_factor=6, strides=1),
        ])
        self.final_stage = ContextEmbeddingBlock(in_channels=128)

    def call(self, inputs):
        stem_output = self.stem(inputs)
        s1_output = self.stage_1(stem_output)
        s2_output = self.stage_2(s1_output)
        s3_output = self.stage_3(s2_output)
        final_output = self.final_stage(s3_output)

        return stem_output, s1_output, s2_output, s3_output, final_output