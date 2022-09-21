#include "./dqn.h"

struct ValueNet : nn::Module
{
    ValueNet(itup& s_shape_tup, itup& a_shape_tup, int hl_sizes[])
        : fc1(std::get<0>(s_shape_tup), hl_sizes[0]),
        fc2(hl_sizes[0], hl_sizes[1]),
        fc3(hl_sizes[1], std::get<0>(a_shape_tup))
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    // Private Functions
    
    // Public Functions
    
    /*
     * Store experienced transition data.
     *
     * Returns: 
     */

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);
        return x;
    }

    nn::Linear fc1, fc2, fc3;
};
TORCH_MODULE(ValueNet);

/*
// ref
struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl(int k_noise_size)
        : conv1(nn::ConvTranspose2dOptions(k_noise_size, 256, 4)
                .bias(false)),
        batch_norm1(256),
        conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                .stride(2)
                .padding(1)
                .bias(false)),
        batch_norm2(128),
        conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                .stride(2)
                .padding(1)
                .bias(false)),
        batch_norm3(64),
        conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                .stride(2)
                .padding(1)
                .bias(false))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch.tanh(conv4(x));

        return x;
    }

    nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator);
*/
