import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort

model = torch.load("/home/zhanghanbo/projects/RCAN/experiment/RCAN-x4/model/model_best.pt",
                       weights_only=False).to('cuda:0')
model.eval()

dummy_input = torch.randn(1, 3, 256, 192).to('cuda:0')

onnx_file_path = "flagged/RCAN_x4.onnx"

print("begin...")
with torch.no_grad():
    torch.onnx.export(
        model,                     
        dummy_input,               
        onnx_file_path,            
        export_params=True,        
        opset_version=9,         
        do_constant_folding=True,   
        input_names=['input'],      
        output_names=['output'],  
        dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'}, #  
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
    )

print(f"模型成功导出为 {onnx_file_path}")

onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)

print("ONNX 模型验证成功！")
