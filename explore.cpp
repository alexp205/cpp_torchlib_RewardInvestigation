#include "./explore.h"

EpsilonMgr::EpsilonMgr (double epsilon_init, double epsilon_decay, double epsilon_min)
{
    epsilon = epsilon_init;
    init = epsilon_init;
    decay = epsilon_decay;
    min_bound = epsilon_min;
}

// Private Functions

// Public Functions

double EpsilonMgr::get()
{
    return epsilon;
}

void EpsilonMgr::decay_step()
{
    double cand_epsilon = epsilon * epsilon_decay;
    epsilon = std::max(min_bound, cand_epsilon);
}

void EpsilonMgr::reset()
{
    epsilon = init;
}
