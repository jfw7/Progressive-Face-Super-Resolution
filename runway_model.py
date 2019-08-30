import runway
import torch
from model import Generator
import torchvision.transforms as transforms
from torch.autograd import Variable as Variable

options = {
    # "batch_size": runway.data_types.number(
    #     min=1,
    #     default=16,
    # ),
    "checkpoint_path": runway.data_types.text(
        default='./checkpoints/generator_checkpoint.ckpt',
    ),
    # "data_path": runway.data_types.text(
    #     default='./dataset/',
    # ),
    # "workers": runway.data_types.number(
    #     min=1,
    #     default=16,
    # ),
}

@runway.setup(options=options)
def setup(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator().to(device)
    g_checkpoint = torch.load(opts['checkpoint_path'], map_location=device)
    generator.load_state_dict(g_checkpoint['model_state_dict'], strict=False)
    return generator

pre_process = transforms.Compose([
    transforms.CenterCrop((178, 178)),
    transforms.Resize((128, 128)),
])

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

_64x64_down_sampling = transforms.Resize((64, 64))
_32x32_down_sampling = transforms.Resize((32, 32))
_16x16_down_sampling = transforms.Resize((16,16))

unloader = transforms.ToPILImage()  # reconvert into PIL image


@runway.command('upscale', inputs={'image': runway.data_types.image}, outputs={'upscaled': runway.data_types.image})
def upscale(model, inputs):
    target_image = pre_process(inputs['image'])
    x4_target_image = _64x64_down_sampling(target_image)
    x2_target_image = _32x32_down_sampling(x4_target_image)
    input_image = _16x16_down_sampling(x2_target_image)

    input_image = Variable(to_tensor(input_image).unsqueeze(0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image.to(device)
    predicted_image = model(input_image, 3, 1)
    upscaled = 0.5 * predicted_image.squeeze(0) + 0.5
    return { 'upscaled': unloader(upscaled) }


if __name__ == '__main__':
    runway.run(port=4231)
