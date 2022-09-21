#include "./agent.h"

// ref: https://github.com/navneet-nmk/Pytorch-RL-CPP

DQNAgent::DQNAgent (torch::Device device, double gamma_hp, double alpha_hp, int mem_len_hp, EpsilonMgr& epsilon_manager, itup& s_shape_tup, itup& a_shape_tup)
{
    std::mt19937 rng(rd());
    gamma = gamma_hp;
    alpha = alpha_hp;
    mem_len = mem_len_hp;
    s_shape = s_shape_tup;
    a_shape = a_shape_tup;

    ValueNet v_net(s_shape_tup, a_shape_tup); // TODO
    torch::optim::Adam optimizer(v_net->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));
    replay_buffer = init_replay_buffer(); // TODO needed?
    epsilon_mgr = epsilon_manager;

    setup_agent(device);
}

// Private Functions

vector<Transition> DQNAgent::init_replay_buffer()
{
    vector<Transition> replay_buf;

    return replay_buf;
}

void DQNAgent::setup_agent(torch::Device device)
{
    v_net->to(device);
}

// Public Functions

/*
 * Store experienced transition data.
 *
 * Returns: 
 */
void DQNAgent::store_experience(Transition& x)
{
    replay_buffer.push_back(x);
}

void DQNAgent::update_models(torch::Device device, int x)
{
    // TODO check
    vector<Transition> batch = .to(device);
    vector<torch::Tensor> states_vec = ;
    vector<torch::Tensor> actions_vec = ;
    vector<torch::Tensor> next_states_vec = ;
    vector<torch::Tensor> rewards_vec = ;
    vector<torch::Tensor> terminals_vec = ;
    for (int i = 0; i < batch.size(); i++) {
        states_vec.push_back(std::get<0>(i));
        actions_vec.push_back(std::get<1>(i));
        next_states_vec.push_back(std::get<2>(i));
        rewards_vec.push_back(std::get<3>(i));
        terminals_vec.push_back(std::get<4>(i));
    }
    torch::Tensor states = torch::cat(state_vec, 0);
    torch::Tensor actions = torch::cat(action_vec, 0);
    torch::Tensor next_states = torch::cat(next_state_vec, 0);
    torch::Tensor rewards = torch::cat(reward_vec, 0);
    torch::Tensor terminals = torch::cat(terminal_vec, 0);

    torch::Tensor Q_vals = v_net->forward(state_tensor);
    torch::Tensor next_Q_target_vals = target_v_net->forward(next_states);
    torch::Tensor next_Q_vals = v_net->forward(next_states);

    torch::Tensor Q_act = Q_vals.gather(1, actions.unsqueeze(1)).squeeze(1);
    torch::Tensor Q_max = std::get<1>(next_Q_vals.max(1));
    torch::Tensor next_Q_act = next_Q_target_vals.gather(1, Q_max.unsqueeze(1)).squeeze(1);
    torch::Tensor Q_y = rewards + gamma * next_Q_vals * (1 - terminals);

    torch::Tensor loss = torch::mse_loss(Q_act, Q_y);

    optimizer->zero_grad();
    loss.backward();
    optimizer->step();

    /*
    // ref
    discriminator->zero_grad();
    torch::Tensor real_imgs = batch.data.to(device);
    torch::Tensor real_lbls = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
    //torch::Tensor real_output = discriminator->forward(real_imgs).reshape({k_batch_size, 1, 1, k_batch_size});
    torch::Tensor real_output = discriminator->forward(real_imgs).reshape({k_batch_size});
    torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_lbls);
    d_loss_real.backward();
    torch::Tensor d_loss = d_loss_real + d_loss_fake;
    discriminator_optimizer.step();

    if (0 == batch_idx % k_log_interval) {
        std::printf("\r[%11d/%11d][%11d/%11d] D_loss: %.4f | G_loss: %.4f\n",
                (long long)epoch,
                (long long)k_num_epochs,
                (long long)batch_idx,
                (long long)batches_per_epoch,
                (double)d_loss.item<float>(),
                (double)g_loss.item<float>());
    }
    */

    epsilon_mgr->decay_step()
}

int DQNAgent::act(vd state)
{
    int action = -1;

    double r = (double) rand() / (RAND_MAX)
    if (r <= epsilon_mgr.get()) {
        std::uniform_int_distribution<int> uni(0,std::get<0>(a_shape));
        action = uni(rng);
    }
    else {
        // https://discuss.pytorch.org/t/how-to-convert-vector-int-into-c-torch-tensor/66539
        auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat16);
        torch::Tensor state_input = torch::from_blob(state.data(), {1, state.size()}, tensor_opts).to(device);
        torch::Tensor Q_val_output = vnet->forward(state_input).reshape({1});

        std::tuple<at::Tensor,at::Tensor> result = torch::max(Q_val_output, 1);
        action = std::get<1>(result);
    }

    std::cout << action << std::endl;

    return action;
}

void DQNAgent::save(int iter, string fname)
{
    // TODO check
    torch::save(v_net, fname+"-v-net-chkpt.pt");
    torch::save(optimizer, fname+"-opt-chkpt.pt");
    std::cout << "\n-> checkpoint iter " << iter << "\n";
}

void DQNAgent::load(int iter, string fname)
{
    // TODO check
    torch::load(v_net, fname+"-v-net-chkpt.pt");
    torch::load(optimizer, fname+"-opt-chkpt.pt");
}
