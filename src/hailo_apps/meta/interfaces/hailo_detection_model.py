import numpy as np

from typing import Any
from pathlib import Path

from contextlib import ExitStack
from collections.abc import Iterator

from PIL import Image
from pydantic import BaseModel, ConfigDict
from hailo_platform import (
    ConfigureParams,
    FormatType,
    HEF,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


class Letterbox(BaseModel):
    model_config = ConfigDict(frozen=True)

    scale: float
    x_offset: int
    y_offset: int


class HailoDetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: list[dict[str, Any]]


class HailoModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    data: dict[str, Any]


class HailoDetectionModel:
    scrfd_anchor_count = 2
    nms_iou_threshold = 0.45

    def __init__(self, model_dir: str, model_name: str):
        self.model_path = Path(model_dir)
        self.model_dir = (
            self.model_path.parent
            if self.model_path.is_file()
            else self.model_path
        )

        self.model_config = self.load_model_config(
            model_path=self.model_path,
            model_name=model_name,
        )

        self.model_config_dir = self.model_config.path.parent
        self.preprocess_config = self.model_config.data["PRE_PROCESS"][0]
        self.postprocess_config = self.model_config.data.get(
            "POST_PROCESS",
            [{}],
        )[0]

        self.input_width = int(self.preprocess_config["InputW"])
        self.input_height = int(self.preprocess_config["InputH"])
        self.quantized_input = bool(
            self.preprocess_config.get("InputQuantEn", True)
        )

        self.output_conf_threshold = float(
            self.postprocess_config.get("OutputConfThreshold", 0.25)
        )

        self.labels = self.load_labels()
        hef_path = self.get_hef_path()

        self.hef = HEF(str(hef_path))
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]

        self.exit_stack = ExitStack()
        self.target = self.exit_stack.enter_context(VDevice())
        configure_params = ConfigureParams.create_from_hef(
            self.hef,
            interface=HailoStreamInterface.PCIe,
        )

        self.network_group = self.target.configure(
            self.hef,
            configure_params,
        )[0]

        self.network_group_params = self.network_group.create_params()
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=self.quantized_input,
            format_type=(
                FormatType.UINT8 if self.quantized_input else FormatType.FLOAT32
            ),
        )

        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        self.exit_stack.enter_context(
            self.network_group.activate(self.network_group_params)
        )

        self.infer_pipeline = self.exit_stack.enter_context(
            InferVStreams(
                self.network_group,
                input_vstreams_params,
                output_vstreams_params,
            )
        )

    def get_hef_path(self) -> Path:
        model_parameters = self.model_config.data["MODEL_PARAMETERS"][0]
        return self.model_config_dir / model_parameters["ModelPath"]

    @staticmethod
    def load_model_config(
        model_path: Path,
        model_name: str,
    ) -> HailoModelConfig:
        if model_path.suffix == ".hef":
            return HailoDetectionModel.get_default_hef_config(
                model_path=model_path
            )

        raise ValueError(f"{model_name} is not a direct HEF model")

    @staticmethod
    def get_default_hef_config(model_path: Path) -> HailoModelConfig:
        return HailoModelConfig(
            path=model_path.with_suffix(".json"),
            data={
                "MODEL_PARAMETERS": [
                    {
                        "ModelPath": model_path.name,
                    }
                ],
                "PRE_PROCESS": [
                    {
                        "InputH": 640,
                        "InputW": 640,
                        "InputQuantEn": True,
                        "InputPadMethod": "letterbox",
                    }
                ],
                "POST_PROCESS": [
                    {
                        "OutputPostprocessType": "SCRFD",
                        "OutputConfThreshold": 0.25,
                    }
                ],
            },
        )

    def load_labels(self) -> dict[str, str]:
        return {"0": "face"}

    def __call__(self, np_image: np.ndarray) -> HailoDetectionResult:
        preprocessed_image, letterbox = self.preprocess(np_image=np_image)
        input_data = {
            self.input_vstream_info.name: np.expand_dims(
                preprocessed_image,
                axis=0,
            ),
        }

        output_data = self.infer_pipeline.infer(input_data)
        return HailoDetectionResult(
            results=list(
                self.parse_results(
                    output_data=output_data,
                    image_width=np_image.shape[1],
                    image_height=np_image.shape[0],
                    letterbox=letterbox,
                )
            )
        )

    def preprocess(self, np_image: np.ndarray) -> tuple[np.ndarray, Letterbox]:
        input_image = Image.fromarray(np_image.astype(np.uint8))
        input_width, input_height = input_image.size

        if self.preprocess_config.get("InputPadMethod") != "letterbox":
            resized_image = input_image.resize(
                (self.input_width, self.input_height)
            )

            return (
                self.format_input(np_image=np.asarray(resized_image)),
                Letterbox(
                    scale=self.input_width / input_width,
                    x_offset=0,
                    y_offset=0,
                ),
            )

        scale = min(
            self.input_width / input_width,
            self.input_height / input_height,
        )

        resized_width = int(input_width * scale)
        resized_height = int(input_height * scale)
        resized_image = input_image.resize((resized_width, resized_height))
        letterboxed_image = Image.new(
            "RGB",
            (self.input_width, self.input_height),
            (114, 114, 114),
        )

        x_offset = (self.input_width - resized_width) // 2
        y_offset = (self.input_height - resized_height) // 2
        letterboxed_image.paste(resized_image, (x_offset, y_offset))

        return (
            self.format_input(np_image=np.asarray(letterboxed_image)),
            Letterbox(
                scale=scale,
                x_offset=x_offset,
                y_offset=y_offset,
            ),
        )

    def format_input(self, np_image: np.ndarray) -> np.ndarray:
        if self.quantized_input:
            return np_image.astype(np.uint8)

        return np_image.astype(np.float32)

    def parse_results(
        self,
        output_data: Any,
        image_width: int,
        image_height: int,
        letterbox: Letterbox,
    ) -> Iterator[dict[str, Any]]:
        if not isinstance(output_data, dict):
            raise TypeError("SCRFD inference output must be a dictionary")

        if not self.is_scrfd_outputs(output_data):
            raise ValueError("Unsupported SCRFD output shape")

        yield from self.parse_scrfd_outputs(
            output_data=output_data,
            image_width=image_width,
            image_height=image_height,
            letterbox=letterbox,
        )

    def is_scrfd_outputs(self, output_data: dict[str, np.ndarray]) -> bool:
        output_channels = sorted(
            self.remove_batch_axis(output).shape[-1]
            for output in output_data.values()
        )

        return output_channels == [2, 2, 2, 8, 8, 8, 20, 20, 20]

    def parse_scrfd_outputs(
        self,
        output_data: dict[str, np.ndarray],
        image_width: int,
        image_height: int,
        letterbox: Letterbox,
    ) -> Iterator[dict[str, Any]]:
        decoded_results = sorted(
            (
                result
                for result in self.decode_scrfd_outputs(
                    output_data=output_data,
                    image_width=image_width,
                    image_height=image_height,
                    letterbox=letterbox,
                )
                if result["score"] >= self.output_conf_threshold
            ),
            key=lambda result: result["score"],
            reverse=True,
        )

        yield from self.nms(results=decoded_results)

    def decode_scrfd_outputs(
        self,
        output_data: dict[str, np.ndarray],
        image_width: int,
        image_height: int,
        letterbox: Letterbox,
    ) -> Iterator[dict[str, Any]]:
        bbox_outputs = {
            output.shape[0]: output.reshape(
                output.shape[0],
                output.shape[1],
                self.scrfd_anchor_count,
                4,
            )
            for output in map(self.remove_batch_axis, output_data.values())
            if output.shape[-1] == self.scrfd_anchor_count * 4
        }

        class_outputs = {
            output.shape[0]: output.reshape(
                output.shape[0],
                output.shape[1],
                self.scrfd_anchor_count,
            )
            for output in map(self.remove_batch_axis, output_data.values())
            if output.shape[-1] == self.scrfd_anchor_count
        }

        for grid_size, bbox_output in bbox_outputs.items():
            class_output = class_outputs[grid_size]
            stride = self.input_width / grid_size
            yield from self.decode_scrfd_grid(
                bbox_output=bbox_output,
                class_output=class_output,
                stride=stride,
                image_width=image_width,
                image_height=image_height,
                letterbox=letterbox,
            )

    def decode_scrfd_grid(
        self,
        bbox_output: np.ndarray,
        class_output: np.ndarray,
        stride: float,
        image_width: int,
        image_height: int,
        letterbox: Letterbox,
    ) -> Iterator[dict[str, Any]]:
        scores = self.sigmoid(class_output)
        y_indexes, x_indexes, anchor_indexes = np.where(
            scores >= self.output_conf_threshold
        )

        for y_index, x_index, anchor_index in zip(
            y_indexes,
            x_indexes,
            anchor_indexes,
        ):
            left, top, right, bottom = (
                bbox_output[y_index, x_index, anchor_index] * stride
            )

            x_center = (x_index + 0.5) * stride
            y_center = (y_index + 0.5) * stride

            yield self.format_result(
                class_id=0,
                bbox=(
                    x_center - left,
                    y_center - top,
                    x_center + right,
                    y_center + bottom,
                ),
                score=float(scores[y_index, x_index, anchor_index]),
                image_width=image_width,
                image_height=image_height,
                letterbox=letterbox,
            )

    @staticmethod
    def remove_batch_axis(output: np.ndarray) -> np.ndarray:
        if output.ndim == 4:
            return output[0]

        return output

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        if np.min(values) >= 0.0 and np.max(values) <= 1.0:
            return values

        positive_values = values >= 0
        negative_values = ~positive_values
        sigmoid_values = np.empty_like(values, dtype=np.float32)

        sigmoid_values[positive_values] = 1.0 / (
            1.0 + np.exp(-values[positive_values])
        )

        exp_values = np.exp(values[negative_values])
        sigmoid_values[negative_values] = exp_values / (1.0 + exp_values)

        return sigmoid_values

    def nms(self, results: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        remaining_results = results

        while remaining_results:
            selected_result = remaining_results[0]
            yield selected_result

            remaining_results = [
                result
                for result in remaining_results[1:]
                if result["category_id"] != selected_result["category_id"]
                or self.iou(result["bbox"], selected_result["bbox"])
                < self.nms_iou_threshold
            ]

    @staticmethod
    def iou(first_bbox: list[float], second_bbox: list[float]) -> float:
        first_x_min, first_y_min, first_x_max, first_y_max = first_bbox
        second_x_min, second_y_min, second_x_max, second_y_max = second_bbox

        intersection_x_min = max(first_x_min, second_x_min)
        intersection_y_min = max(first_y_min, second_y_min)
        intersection_x_max = min(first_x_max, second_x_max)
        intersection_y_max = min(first_y_max, second_y_max)

        intersection_width = max(0.0, intersection_x_max - intersection_x_min)
        intersection_height = max(0.0, intersection_y_max - intersection_y_min)
        intersection_area = intersection_width * intersection_height

        first_area = (first_x_max - first_x_min) * (first_y_max - first_y_min)
        second_area = (second_x_max - second_x_min) * (
            second_y_max - second_y_min
        )

        union_area = first_area + second_area - intersection_area
        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    def format_result(
        self,
        class_id: int,
        bbox: tuple[float, float, float, float],
        score: float,
        image_width: int,
        image_height: int,
        letterbox: Letterbox,
    ) -> dict[str, Any]:
        x_min, y_min, x_max, y_max = bbox

        if max(bbox) <= 1.5:
            x_min *= self.input_width
            x_max *= self.input_width
            y_min *= self.input_height
            y_max *= self.input_height

        return {
            "bbox": [
                self.scale_x(
                    x=x_min,
                    image_width=image_width,
                    letterbox=letterbox,
                ),
                self.scale_y(
                    y=y_min,
                    image_height=image_height,
                    letterbox=letterbox,
                ),
                self.scale_x(
                    x=x_max,
                    image_width=image_width,
                    letterbox=letterbox,
                ),
                self.scale_y(
                    y=y_max,
                    image_height=image_height,
                    letterbox=letterbox,
                ),
            ],
            "score": float(score),
            "category_id": class_id,
            "label": self.labels.get(str(class_id), f"class_{class_id}"),
        }

    @staticmethod
    def scale_x(x: float, image_width: int, letterbox: Letterbox) -> float:
        scaled_x = (x - letterbox.x_offset) / letterbox.scale
        return max(0.0, min(scaled_x, image_width))

    @staticmethod
    def scale_y(y: float, image_height: int, letterbox: Letterbox) -> float:
        scaled_y = (y - letterbox.y_offset) / letterbox.scale
        return max(0.0, min(scaled_y, image_height))

    def close(self) -> None:
        self.exit_stack.close()

    def __del__(self) -> None:
        if hasattr(self, "exit_stack"):
            self.close()
