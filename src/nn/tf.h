#ifndef TF_H
#define TF_H

#include <functional>
#include <cmath>

// transfer functions
struct TF {
  enum class TYPE {LINEAR,SIGMOID,TANH,RELU,RELU_LEAKY};
  typedef std::function<double(double)> tf_func;
  static std::function<double(double)> get_activate(const TYPE type)
  {
    switch (type) {
      case TYPE::LINEAR:return [](double x){return x;};
      case TYPE::SIGMOID:return [](double x){return 1.0/(1.0+exp(-x));};
      case TYPE::TANH:return [](double x){return tanh(x);};
      case TYPE::RELU:return [](double x){return std::max(0.0,x);};
      case TYPE::RELU_LEAKY:return [](double x){return (x<0)?0.01*x:x;};
      default:return nullptr;
    }
  }
  static std::function<double(double)> get_grad(const TYPE type)
  {
    switch (type) {
      case TYPE::LINEAR:return [](double){return 1.0;};
      case TYPE::SIGMOID:return [](double x){return x*(1.0-x);};
      case TYPE::TANH:return [](double x){return 1.0-x*x;};
      case TYPE::RELU:return [](double x){return x>0?1:0;};
      case TYPE::RELU_LEAKY:return [](double x){return (x<0)?0.01:1.0;};
      default:return nullptr;
    }
  }
};

typedef std::function<double(double)> tf_func;

#endif // TF_H
