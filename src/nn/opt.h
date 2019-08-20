#ifndef OPT_H
#define OPT_H

#include <cmath>
#include <memory>
#include <cassert>

// Stochastic gradient descent class
class OPT {
  public:
    enum class TYPE {SGD,ADAM,NADAM};
    virtual void UpdateWeights(double alpha,const vec2D &G,vec2D &w)
    {
      for (size_t j=0;j<w.size();j++)
        for (size_t i=0;i<w[0].size();i++)
          w[j][i]+=-alpha*G[j][i];
    }
    virtual ~OPT() {
    }
};

class SGD_ADAM : public OPT {
  public:
    SGD_ADAM(int input_units,int output_units)
    :S(output_units,vec1D(input_units)),M(output_units,vec1D(input_units))
    {
      beta1=0.9;
      power_beta1=beta1;
      beta2=0.999;
      power_beta2=beta2;
      eps=1E-8;
    }
    void UpdateWeights(double alpha,const vec2D &G,vec2D &w)
    {
      assert( (M.size()==G.size()) && (M[0].size()==G[0].size()));
      const size_t nsize=w.size();
      const size_t isize=w[0].size();
      for (size_t n=0;n<nsize;n++) {
        for (size_t i=0;i<isize;i++) {
          M[n][i]=beta1*M[n][i]+(1.0-beta1)*(G[n][i]);
          S[n][i]=beta2*S[n][i]+(1.0-beta2)*(G[n][i]*G[n][i]);

          const double Mnorm=M[n][i]/(1.0-power_beta1);
          const double Snorm=S[n][i]/(1.0-power_beta2);

          w[n][i]+=-alpha*Mnorm/(sqrt(Snorm)+eps);
        }
      }
      power_beta1*=beta1;
      power_beta2*=beta2;
    }
  protected:
    vec2D S,M;
    double beta1,beta2,power_beta1,power_beta2,eps;
};

class SGD_NADAM : public OPT {
  public:
    SGD_NADAM(int input_units,int output_units)
    :S(output_units,vec1D(input_units)),M(output_units,vec1D(input_units))
    {
      beta1=0.9;
      beta2=0.999;
      power_beta1=power_beta2=1.0;
      power_beta11=beta1;
      eps=1E-8;
    }
    void UpdateWeights(double alpha,const vec2D &G,vec2D &w)
    {
      power_beta1*=beta1;
      power_beta11*=beta1;
      power_beta2*=beta2;
      assert( (M.size()==G.size()) && (M[0].size()==G[0].size()));
      const size_t nsize=w.size();
      const size_t isize=w[0].size();
      for (size_t n=0;n<nsize;n++) {
        for (size_t i=0;i<isize;i++) {
          M[n][i]=beta1*M[n][i]+(1.0-beta1)*(G[n][i]);
          S[n][i]=beta2*S[n][i]+(1.0-beta2)*(G[n][i]*G[n][i]);

          double m_hat=beta1*M[n][i]/(1.0-power_beta11)+((1.0-beta1)*G[n][i]/(1.0-power_beta1));
          double n_hat=beta2*S[n][i]/(1.0-power_beta2);
          w[n][i]-=alpha*m_hat/(sqrt(n_hat)+1E-8);
        }
      }
    }
  protected:
    vec2D S,M;
    double beta1,beta2,power_beta1,power_beta11,power_beta2,eps;
};


#endif // OPT_H
