#ifndef NN_H
#define NN_H

#include "matrix.h"
#include "nnutil.h"
#include "tf.h"
#include <iostream>

// Basic dense layer
class LayerDense {
  public:
    LayerDense(int input_units,int output_units,TF::TYPE tf,OPT::TYPE to,double alpha,std::vector <double*> &gptr,std::vector <double*> &wptr,bool cbias)
    :input_units_(input_units),output_units_(output_units),alpha_(alpha),bias_(cbias)
    {
      output.resize(output_units);
      wx.resize(output_units,vec1D(input_units));
      Gx.resize(output_units,vec1D(input_units));
      delta.resize(output_units);
      bias.resize(output_units);
      tf_activate=TF::get_activate(tf);
      tf_grad=TF::get_grad(tf);
      opt_wx=NNUtil::CreateOptimizer(input_units,output_units,to);

      matrix::unfold(wx,wptr);
      matrix::unfold(Gx,gptr);
      if (bias_) {
        matrix::unfold(bias,wptr);
        matrix::unfold(delta,gptr);
      }
    }
    void Forward(const vec1D &input)
    {
      if (bias_) {
        for (int n=0;n<output_units_;n++)
          output[n]=tf_activate(matrix::dot(wx[n],input)+bias[n]);
      } else {
        for (int n=0;n<output_units_;n++)
          output[n]=tf_activate(matrix::dot(wx[n],input));
      }
    }
    void UpdateGradient(const vec1D &input) {
      CalcGradient(input,delta,Gx);
    }
    void UpdateWeights() {
      opt_wx->UpdateWeights(alpha_,Gx,wx);
      if (bias_) NNUtil::UpdateWeightsClassic(alpha_,delta,bias);
    };
    void SoftmaxOutput() {
      double sum=0.0;
      for (int n=0;n<output_units_;n++) sum+=exp(output[n]);
      for (int n=0;n<output_units_;n++) output[n]=exp(output[n])/sum;
    }

    void DeltaOutputL2(const vec1D &target)
    {
      assert((target.size()==output.size()) && (target.size()==delta.size()));
      for (int n=0;n<output_units_;n++)
        delta[n]=(output[n]-target[n])*tf_grad(output[n]);
    }
    /*void DeltaOutputRes()
    {
      for (int n=0;n<output_units_;n++)
        delta[n]=-tf_grad(output[n]);
    }*/
    double NextDelta(int n) const {
      return matrix::dot_row_col(delta,wx,n);
    }
    void Delta(const LayerDense *next_layer)
    {
      for (int n=0;n<output_units_;n++)
        delta[n]=next_layer->NextDelta(n)*tf_grad(output[n]);
    }
    void InitWeights(double scale,std::mt19937 &mt) {
      NNUtil::InitWeights(scale,mt,wx);
    }
    void CalcGradient(const vec1D &input,const vec1D &delta,vec2D &G)
    {
      assert((G.size()==delta.size()) && (G[0].size()==input.size()));
      for (size_t n=0;n<G.size();n++) {
        for (size_t i=0;i<G[0].size();i++) {
          G[n][i]=delta[n]*input[i]; // gradient of error function
        }
      }
    }
    int input_units_,output_units_;
    vec1D output;
    vec2D wx,Gx;
    vec1D delta,bias;
    TF::tf_func tf_activate,tf_grad;
    double alpha_,bias_;
    std::unique_ptr<OPT> opt_wx;
};


class NN_MLP {
  public:
    NN_MLP(int input_units,OPT::TYPE opt_type=OPT::TYPE::ADAM,double alpha=0.001,double scale=0.01)
    :x(input_units),mt(0),opt_type_(opt_type),alpha_(alpha),scale_(scale),softmax(false)
    {

    }
    void SetSoftMax(bool b)
    {
      softmax=b;
    }
    int NumWeights() {
      return wptr.size();
    }
    void AddLayer(int output_units,TF::TYPE tf_type,bool bias=true)
    {
      int input_units=layer.size()?layer.back()->output.size():x.size();
      layer.push_back(new LayerDense(input_units,output_units,tf_type,opt_type_,alpha_,gptr,wptr,bias));
      layer.back()->InitWeights(scale_,mt);
    }
    ~NN_MLP() {
      for (auto l:layer) delete l;
    }
    const vec1D &Output() const {return layer.back()->output;};
    void Predict(const vec1D &x_in) // forward pass
    {
      x=x_in;
      layer[0]->Forward(x);
      for (size_t i=1;i<layer.size();i++)
        layer[i]->Forward(layer[i-1]->output);
      if (softmax) layer.back()->SoftmaxOutput();
    }
    /*void CalcGradients() {
      layer.back()->DeltaOutputRes();
      CalcGradientsBackward();
    }*/
    void CalcGradients(const vec1D &target) {
      layer.back()->DeltaOutputL2(target);
      CalcGradientsBackward();
    }
    void Update(const vec1D &target)
    {
      CalcGradients(target);
      UpdateWeights();
    }
    /*double CheckGradient(const vec1D &input,const vec1D &target,bool verbose=false,double eps=1E-5)
    {
      std::vector <double>vg,vt;
      Predict(input);
      CalcGradients(target);
      for (auto l:layer) {
        for (int i=0;i<l->NumWeights();i++) {
          double d=CalcNumericalDerivativeL2(input,target,l->GetWeight(i),eps);
          if (verbose) std::cout << d << " " << *(l->GetGrad(i)) << std::endl;
          vg.push_back(*l->GetGrad(i));
          vt.push_back(d);
        }
      }
      return sqrt(norm2(vg-vt))/(sqrt(norm2(vg+vt)));
      return -1;
    }*/
    std::vector <double*> wptr; // ptr to all weights
    std::vector <double*> gptr; // ptr to all gradients
  private:
    void CalcGradientsBackward() {
      for (size_t i=layer.size()-1;i>0;i--)
        layer[i-1]->Delta(layer[i]);

      layer[0]->UpdateGradient(x);
      for (size_t i=1;i<layer.size();i++)
        layer[i]->UpdateGradient(layer[i-1]->output);
    }
    void UpdateWeights()
    {
      for (auto l:layer) l->UpdateWeights();
    }
    /*double CalcNumericalDerivativeL2(const vec1D &input,const vec1D &target,double *pw,double eps)
    {
      double wo=*pw;
      *pw=wo+eps;
      Predict(input);
      double t0=0.5*norm2(target-Output()); // Calc J(x+h)
      *pw=wo-eps;
      Predict(input);
      double t1=0.5*norm2(target-Output());// Calc J(x-h)
      *pw=wo; // restore weight

      return (t0-t1)/(2*eps); // return approx. gradient as central difference
    }*/
    vec1D x;
    std::mt19937 mt;
    OPT::TYPE opt_type_;
    double alpha_,scale_;
    std::vector<LayerDense*> layer;
    bool softmax;
};

#endif // NN_H
