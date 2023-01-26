import tensorflow as tf
from nn.layers.BiseNetV2_layers.SemanticBranch import SemanticBranch
from nn.layers.BiseNetV2_layers.DetailBranch import DetailBranch
from nn.layers.BiseNetV2_layers.SegmentHead import SegmentHead
from nn.layers.BiseNetV2_layers.BilateralGuidedAggregation import BilateralGuidedAggregation

class BiseNetV2_training_ver(tf.keras.Model):
    def __init__(self, n_classes, dims):
        # dims is a 1d list of 2 numbers, those being height and width of image
        super(BiseNetV2_training_ver, self).__init__()
        self.semantic = SemanticBranch()
        self.details = DetailBranch()
        self.stem_boost = SegmentHead(in_channels=16, n_classes=n_classes, size=dims)
        self.s1_boost = SegmentHead(in_channels=32, n_classes=n_classes, size=dims)
        self.s2_boost = SegmentHead(in_channels=64, n_classes=n_classes, size=dims)
        self.s3_boost = SegmentHead(in_channels=128, n_classes=n_classes, size=dims)
        self.aggregation_layer = BilateralGuidedAggregation()
        self.output_head = SegmentHead(in_channels=128, n_classes=n_classes, size=dims)

    def call(self, inputs):
        stem_output, s1_output, s2_output, s3_output, final_output = self.semantic(inputs)
        stem_output = self.stem_boost(stem_output)
        s1_final = self.s1_boost(s1_output)
        s2_final = self.s2_boost(s2_output)
        s3_final = self.s3_boost(s3_output)
        details = self.details(inputs)
        combined = self.aggregation_layer([details, final_output])

        combined_output = self.output_head(combined)

        return stem_output, s1_final, s2_final, s3_final, combined_output