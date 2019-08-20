#ifndef NNUTIL_H
#define NNUTIL_H

#include "opt.h"

#include <functional>
#include <random>
#include <memory>

class NNUtil {
  public:
    /*static void GradientAdd(const vec1D &delta,vec1D &w)
    {
      assert( (delta.size()==w.size()));
      #if 1
        for (size_t n=0;n<w.size();n++) w[n]+=(delta[n]);
      #else
        std::transform(begin(w),end(w),begin(delta),begin(w),std::plus<double>());
      #endif
    }
    static void GradientAdd(const vec1D &input,const vec1D &delta,vec2D &w)
    {
      assert( (delta.size()==w.size()) && (input.size()==w[0].size()));
      for (size_t n=0;n<w.size();n++)
        for (size_t i=0;i<w[0].size();i++) {
          w[n][i]+=(delta[n]*input[i]);
        }
    }*/
    static void InitWeights(double scale,std::mt19937 &mt,vec2D &w)
    {
      /*int fan_out=(int)w.size();
      int fan_in=(int)w[0].size();
      const double a=sqrt(6.0/double(fan_in+fan_out)); // glorot-initalization*/
      std::uniform_real_distribution<double> dist(-scale,scale);
      for (size_t j=0;j<w.size();j++)
        for (size_t i=0;i<w[0].size();i++)
        {
          w[j][i]=dist(mt);
        }
    }
    static void UpdateWeightsClassic(double alpha,const vec1D &G,vec1D &weights)
    {
      assert(G.size()==weights.size());
      for (size_t i=0;i<weights.size();i++)
        weights[i]+=-alpha*G[i];
    }
    static std::unique_ptr<OPT> CreateOptimizer(int input_units,int output_units,OPT::TYPE type)
    {
      switch (type) {
        case OPT::TYPE::SGD: return std::make_unique<OPT>();
        case OPT::TYPE::ADAM: return std::make_unique<SGD_ADAM>(input_units,output_units);
        case OPT::TYPE::NADAM: return std::make_unique<SGD_NADAM>(input_units,output_units);
        default: return nullptr;
      }
    }
};

#endif // NNUTIL_H
