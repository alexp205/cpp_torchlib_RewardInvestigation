#pragma once

#ifndef AGENT_H_
#define AGENT_H_

#include "./dqn.h"
#include <vector>
#include <tuple>

typedef vector<double> vd;
typedef tuple<int> itup;

struct Transition
{
    vd state;
    int action;
    double reward;
    vd next_state;
    bool done;
};

class DQNAgent
{
    std::random_device rd;
    double gamma;
    double alpha;
    int mem_len;
    itup s_shape;
    itup a_shape;
    vector<Transition> replay_buffer;
    EpsilonMgr& epsilon_mgr;

    vector<Transition> init_replay_buffer();
    void setup_agent(torch::Device);

    public:
    DQNAgent(torch::Device, double, double, int, EpsilonMgr&, itup&, itup&);
    //DQNAgent(const DQNAgent&);
    //DQNAgent& operator=(const DQNAgent&);
    //~DQNAgent();
    void store_experience(int);
    void update_models(int);
    int act(int);
    void print();
};

#endif
