#ifndef OLS_H
#define OLS_H

#include <numeric>
#include <vector>
#include <cmath>

// general linear model of the form p=w1*p1+w2*p2+...+wn*pn
template <class T>
class OLS {
  typedef std::vector<T> vec1D;
  typedef std::vector<vec1D> vec2D;
  const T ftol=1E-8;
  public:
    OLS(int n,int kmax=1,T lambda=0.998,T nu=0.001)
    :n(n),kmax(kmax),lambda(lambda),nu(nu),
    x(n),w(n),b(n),mcov(n,vec1D(n)),mchol(n,vec1D(n))
    {
      km=0;
    }
    T Predict(const vec1D &p) {
      x=p;
      return std::inner_product(begin(x),end(x),begin(w),0.0);
    }
    void Update(T val)
    {
      for (int j=0;j<n;j++)
        for (int i=0;i<n;i++) mcov[j][i]=lambda*mcov[j][i]+(1.0-lambda)*(x[j]*x[i]);

      for (int i=0;i<n;i++) b[i]=lambda*b[i]+(1.0-lambda)*(x[i]*val);

      km++;
      if (km>=kmax) {
        if (!Factor(mcov)) Solve(b,w);
        km=0;
      }
    }
  private:
    int Factor(const vec2D &mcov)
    {
      mchol=mcov; // copy the matrix
      for (int i=0;i<n;i++) mchol[i][i]+=nu;
      for (int i=0;i<n;i++) {
        for (int j=0;j<i;j++) {
          T sum=mchol[i][j];
          for (int k=0;k<j;k++) sum-=(mchol[i][k]*mchol[j][k]);
          mchol[i][j]=sum/mchol[j][j];
        }
        T sum=mchol[i][i];
        for (int k=0;k<i;k++) sum-=(mchol[i][k]*mchol[i][k]);
        if (sum>ftol) mchol[i][i]=sqrt(sum);
        else return 1; // matrix indefinit
      }
      return 0;
    }

    void Solve(const vec1D &b,vec1D &sol)
    {
      for (int i=0;i<n;i++) {
        T sum=b[i];
        for (int j=0;j<i;j++) sum-=(mchol[i][j]*sol[j]);
        sol[i]=sum/mchol[i][i];
      }
      for (int i=n-1;i>=0;i--) {
        T sum=sol[i];
        for (int j=i+1;j<n;j++) sum-=(mchol[j][i]*sol[j]);
        sol[i]=sum/mchol[i][i];
      }
    }
    int n,kmax,km;
    T lambda,nu;
    vec1D x,w,b;
    vec2D mcov,mchol;
};


#endif
