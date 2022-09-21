#pragma once

#ifndef EXPLORE_H_
#define EXPLORE_H_

class EpsilonMgr
{
    double epsilon;
    double init;
    double decay;
    double min_bound;

    public:
    EpsilonMgr(double, double, double);
    //EpsilonMgr(const EpsilonMgr&);
    //EpsilonMgr& operator=(const EpsilonMgr&);
    //~EpsilonMgr();
    double get();
    void decay_step();
    void reset();
}

#endif
