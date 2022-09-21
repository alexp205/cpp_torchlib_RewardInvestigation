#include "./main.h"

int main(int argc, char* argv[])
{
    // TODO check
    torch::Tensor tensor = torch::rand({2,3});
    std::cout << tensor << std::endl;

    torch::manual_seed(1);
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    std::cout << "- parsing args" << std::endl;
    int seed = std::stoi(argv[1]);
    string dataf_name = argv[2];
    bool load;
    argv[3] >> std::boolalpha >> load;
    int batch_size = std::stoi(argv[4]);
    int ep_lim = std::stoi(argv[5]);
    int log_period = std::stoi(argv[6]);
    // DQN specific
    int mem_len = std::stoi(argv[7]);
    int explore_steps = std::stoi(argv[8]);
    double gamma = std::stod(argv[9]);
    double alpha = std::stod(argv[10]);
    double tau = std::stod(argv[11]);
    double epsilon = std::stod(argv[12]);
    double epsilon_decay = std::stod(argv[13]);
    double epsilon_min = std::stod(argv[14]);
    EpsilonMgr epsilom_mgr(epsilon, epsilon_decay, epsilon_min);
    //EpsilonMgr::EpsilonMgr (double epsilon_init, double epsilon_decay, double epsilon_min)

    std::cout << "- setting up env" << std::endl;
    // TODO check
    rlly::env::CartPole env;
    env.set_seed(seed);
    itup s_shape(env.observation_space.n);
    itup a_shape(env.action_space.n);

    std::cout << "- setting up agent" << std::endl;
    DQNAgent agent(device, gamma, alpha, mem_len, epsilon_mgr, s_shape, a_shape);
    //DQNAgent::DQNAgent (torch::Device device, double gamma_hp, double alpha_hp, int mem_len_hp, EpsilonMgr& epsilon_manager, itup& s_shape_tup, itup& a_shape_tup)
    if (load) {
        agent.load();
    }

    std::cout << "- doing exploration" << std::endl;
    int a = env.action_space.sample();
    std::cout << "checking action: " << a << std::endl;
    vd s = env.reset();
    for (int i = 0; i < explore_steps; i++) {
        //a = agent.act(s);
        a = env.action_space.sample();
        StepResult<vd> trans = env.step(a);
        std::cout << "state: ";
        rlly::utils::vec::printvec(trans.next_state);
        std::cout << "reward: " << trans.reward << std::endl;

        Transition trans_obj;
        trans_obj.state = s;
        trans_obj.action = a;
        trans_obj.reward = trans.reward;
        trans_obj.next_state = trans.next_state;
        trans_obj.done = trans.done;
        agent.store_experience(trans_obj);

        s = trans.next_state;

        if (trans.done) {
            s = env.reset();
        }
    }
    s = env.reset();

    std::cout << "- doing training" << std::endl;
    int ep = 0;
    int run = 0;
    double ep_r = 0.;
    while (ep < ep_lim) {
        a = agent.act(s);
        StepResult<vd> trans = env.step(a);

        run++;
        ep_r += trans.reward;

        Transition trans_obj;
        trans_obj.state = s;
        trans_obj.action = a;
        trans_obj.reward = trans.reward;
        trans_obj.next_state = trans.next_state;
        trans_obj.done = trans.done;
        agent.store_experience(trans_obj);

        //agent.update_models_target(tau); TODO need to figure out how to do this
        agent.update_models(batch_size);

        s = trans.next_state;

        if (trans.done) {
            std::cout << "episode: " << ep << ", episode reward: " << trans.reward << std::endl;
            ep++;
            s = env.reset();
            ep_r = 0.
        }
    }
}
