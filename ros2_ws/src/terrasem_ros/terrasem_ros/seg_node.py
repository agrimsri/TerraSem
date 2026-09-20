#!/usr/bin/env python3
"""seg_node.py — ROS 2 Python node running ONNX INT8 SegFormer-B0 inference.

Subscribes:
    - /pylon_camera_node/image_raw (sensor_msgs/Image)
Publishes:
    - /terrasem/semantic_image (sensor_msgs/Image, mono8 class IDs)
    - /terrasem/semantic_confidence (sensor_msgs/Image, 32FC1 max softmax prob)
"""

from __future__ import annotations

import os
import time

import numpy as np

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # type: ignore

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class SegNode(Node):  # type: ignore
    def __init__(self) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 (rclpy, cv_bridge) is not available in the current environment.")

        super().__init__("seg_node")
        self.declare_parameter("onnx_model", "checkpoints/segformer_b0_int8_static.onnx")
        self.declare_parameter("input_size", [512, 512])
        self.declare_parameter("intra_threads", 4)

        model_path = self.get_parameter("onnx_model").get_parameter_value().string_value
        if not os.path.exists(model_path):
            # Fallback to dynamic or FP32
            alt_path = "checkpoints/segformer_b0_int8_dynamic.onnx"
            model_path = alt_path if os.path.exists(alt_path) else "checkpoints/segformer_b0.onnx"

        self.get_logger().info(f"Loading ONNX model from: {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.get_parameter("intra_threads").get_parameter_value().integer_value
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/pylon_camera_node/image_raw", self.image_callback, 10)
        self.pub_mask = self.create_publisher(Image, "/terrasem/semantic_image", 10)
        self.pub_conf = self.create_publisher(Image, "/terrasem/semantic_confidence", 10)
        self.get_logger().info("SegNode initialized successfully.")

    def image_callback(self, msg: Image) -> None:
        t0 = time.perf_counter()
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        orig_h, orig_w = cv_img.shape[:2]

        # Preprocess
        import cv2
        resized = cv2.resize(cv_img, (512, 512)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (resized - mean) / std
        input_tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # Inference
        logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0][0]  # [11, 512, 512]

        # Softmax & Argmax
        exp_l = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=0, keepdims=True)
        conf = np.max(probs, axis=0).astype(np.float32)
        preds = np.argmax(logits, axis=0).astype(np.uint8)

        # Resize back to original
        pred_full = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        conf_full = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Publish
        mask_msg = self.bridge.cv2_to_imgmsg(pred_full, encoding="mono8")
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        conf_msg = self.bridge.cv2_to_imgmsg(conf_full, encoding="32FC1")
        conf_msg.header = msg.header
        self.pub_conf.publish(conf_msg)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().debug(f"Segmentation inference latency: {dt_ms:.1f} ms")


def main(args=None) -> None:
    if not ROS2_AVAILABLE:
        print("ROS 2 is not available on host. Use the Docker container to run ROS 2 nodes.")
        return
    rclpy.init(args=args)
    node = SegNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
