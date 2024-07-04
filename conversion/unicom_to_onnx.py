import onnx
import onnx.checker
import torch
from oml.models import ViTUnicomExtractor
import oml

print(f"Oml version: {oml.__version__}")
print(f"Torch version: {torch.__version__}")

model_name = "vitb16_unicom"
onnx_path = f"{model_name}.onnx"
model = ViTUnicomExtractor(weights=model_name,
                           arch=model_name,
                           use_gradiend_ckpt=False,
                           normalise_features=False)
model = model.eval()

dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch_size"},
                          "output": {0: "batch_size"}},
        )

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)