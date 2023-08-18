import matplotlib.pyplot as plt
import PIL
import torch
import torchvision

import mnist_nn

def main():

    device = "cuda"

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(0.5, 1)
        ]
    )

    training_data = torchvision.datasets.MNIST(
        mnist_nn.DOWNLOAD_PATH, True, transform, download=True)

    param = mnist_nn.Parameters(8, 8, 69, 0.017294, 0.021476)
    loss = mnist_nn.get_loss_func()
    training_loader = torch.utils.data.DataLoader(training_data,
                                                  batch_size=mnist_nn.N_BATCH)
    net = mnist_nn.train_model(device, training_loader, param, loss)

    errn_input = PIL.Image.open(mnist_nn.ERRN_FILE)
    errn_input = PIL.ImageOps.grayscale(errn_input)
    errn_input = transform(errn_input)
    errn_input = errn_input[None, :, :, :]
    errn_input = errn_input.to(device)

    with torch.no_grad():
        images = net._conv_layer(errn_input)
        images = images.cpu()

    for i in range(param.n_conv_layer):
        mnist_nn.plot_datapoint(images[0, i], f"header_{i}.png")


if __name__ == "__main__":
    main()
