import io
import torch
import torch.onnx

from lib.utils.config_parse import cfg_from_file
from lib.utils.config_parse import cfg
from lib.modeling.model_builder import create_model


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    confg_file = '/home/hby/mycode/ssds.pytorch-1/experiments/cfgs/my_ssdlite_mobilenetv2_6.yml'
    cfg_from_file(confg_file)

    model, priors = create_model(cfg.MODEL)
        # Utilize GPUs for computation
    use_gpu = torch.cuda.is_available()
    #half = False
    if use_gpu:
        print('Utilize GPUs for computation')
        print('Number of GPU available', torch.cuda.device_count())
        model.cuda()
        # self.model = torch.nn.DataParallel(self.model).module
        # Utilize half precision
        half = cfg.MODEL.HALF_PRECISION
        if half:
            model = model.half()

    pthfile = r'/home/hby/mycode/ssds.pytorch-1/experiments/models/ssd_mobilenet_v2_voc_6/ssd_lite_mobilenet_v2_voc_epoch_400.pth'
    checkpoint = torch.load(pthfile, map_location='cuda' if use_gpu else 'cpu')
    model.load_state_dict(checkpoint)
    
    #data type nchw
    dummy_input1 = torch.randn(1, 3, 300, 300).cuda().half()
    # dummy_input2 = torch.randn(1, 3, 64, 64)
    # dummy_input3 = torch.randn(1, 3, 64, 64)
    input_names = [ "actual_input_1"]
    output_names = [ "output1" ]
    # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model, dummy_input1, "ssdmobilenet_half.onnx", verbose=True, input_names=input_names, output_names=output_names)

if __name__ == "__main__":
	test()
