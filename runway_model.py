import runway
import torch
from model import Generator
import torchvision.transforms as transforms
from torch.autograd import Variable as Variable
from dataloader import CelebDataSet
from torch.utils.data import DataLoader
from eval import test
from torch import optim, nn
from torchvision import utils

options = {
    # "batch_size": runway.data_types.number(
    #     min=1,
    #     default=16,
    # ),
    "checkpoint_path": runway.data_types.text(
        default='./checkpoints/unalign_trained_generator_checkpoint.ckpt',
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
    step = g_checkpoint['step']
    alpha = g_checkpoint['alpha']
    iteration = g_checkpoint['iteration']

    print('pre-trained model is loaded step:%d, iteration:%d'%(step, iteration))
    MSE_Loss = nn.MSELoss()

    return (generator, MSE_Loss, step, alpha)\

@runway.command('upscale', inputs={'image': runway.data_types.image, 'iterations': runway.data_types.number(default=1, min=1, max=4)}, outputs={'upscaled': runway.data_types.image})
def upscale(model, inputs):
    (generator, MSE_Loss, step, alpha) = model

    image = inputs['image']

    for i in range(0, inputs['iterations']):
        dataset = CelebDataSet(image = image, state='test')
        dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)


        image = test(dataloader, generator, MSE_Loss, step, alpha).squeeze(0).cpu().detach().cpu()
        image = (image - image.min()) / (image.max() - image.min())
        image = transforms.ToPILImage()(image)


    return { 'upscaled': image }


if __name__ == '__main__':
    runway.run(port=4231)
