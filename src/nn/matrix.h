#ifndef MATRIX_H
#define MATRIX_H

#include "../global.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>
#include <cassert>

// helper functions
class matrix {
  public:
  static void unfold(vec2D &m,std::vector<double*>&x)
  {
    for (size_t j=0;j<m.size();j++) {
      for (size_t i=0;i<m[0].size();i++) {
        x.push_back(&m[j][i]);
      }
    }
  }
  static void unfold(vec1D &v,std::vector<double*>&x)
  {
    for (size_t i=0;i<v.size();i++) {
        x.push_back(&v[i]);
    }
  }

  /*vec1D operator+(const vec1D &v1,const vec1D &v2)
  {
    assert(v1.size()==v2.size());
    vec1D v(v1.size());
    //for (size_t i=0;i<v1.size();i++) v[i]=v1[i]+v2[i];
    std::transform(begin(v1),end(v1),begin(v2),begin(v),std::plus<double>());
    return v;
  }
  vec1D operator-(const vec1D &v1,const vec1D &v2)
  {
    assert(v1.size()==v2.size());
    vec1D v(v1.size());
    //for (size_t i=0;i<v1.size();i++) v[i]=v1[i]-v2[i];
    std::transform(begin(v1),end(v1),begin(v2),begin(v),std::minus<double>());
    return v;
  }
  double operator*(const vec1D &v1,const vec1D &v2)
  {
    assert(v1.size()==v2.size());
    double sum=0.0;
    for (size_t i=0;i<v1.size();i++) sum+=v1[i]*v2[i];
    return sum;
  }
  inline double norm2(const vec1D &v)
  {
    //double sum2=std::accumulate(begin(v),end(v),0.0,[](auto a,auto b){return a+b*b;});
    //for (size_t i=0;i<v.size();i++) sum+=v[i]*v[i];
    return std::inner_product(begin(v),end(v),begin(v),0.0);
  }
  vec1D hadamard(const vec1D &v1,const vec1D &v2)
  {
    assert(v1.size()==v2.size());
    vec1D v(v1.size());
    for (size_t i=0;i<v1.size();i++) v[i]=v1[i]*v2[i];
    return v;
  }*/
  // inner product of v1 and v2
  static double dot(const vec1D &v1,const vec1D &v2)
  {
    assert(v1.size()==v2.size());
    //double sum=0.0;
    //for (size_t i=0;i<v1.size();i++) sum+=v1[i]*v2[i];
    return std::inner_product(begin(v1),end(v1),begin(v2),0.0);;
  }

  vec1D mmul(const vec2D &m,const vec1D &v)
  {
    assert(m[0].size()==v.size());
    vec1D vout(m.size());
    for (size_t j=0;j<m.size();j++) vout[j]=dot(m[j],v);
    return vout;
  }

  vec2D transpose(const vec2D &matrix)
  {
    vec2D mt(matrix[0].size(),vec1D(matrix.size()));
    for (size_t j=0;j<matrix.size();j++)
      for (size_t i=0;i<matrix[0].size();i++) {
        mt[i][j]=matrix[j][i];
      }
    return mt;
  }

  // inner product of vrow with column ncol of m, equivalent to the ncol-th row of m^T*v
  static double dot_row_col(const vec1D &vrow,const vec2D &m,int ncol)
  {
    assert(vrow.size()==m.size());
    double sum=0.0;
    for (size_t i=0;i<vrow.size();i++) sum+=vrow[i]*m[i][ncol];
    return sum;
  }
  void ClearMatrix(vec1D &v)
  {
    std::fill(begin(v),end(v),0.0);
  }
  void ClearMatrix(vec2D &m)
  {
    for (size_t n=0;n<m.size();n++)
      ClearMatrix(m[n]);
  }

  // v_out+=alpha*v_in
  /*void vec_add(const vec1D &v_in,double alpha,vec1D &v_out) {
    assert(v_in.size()==v_out.size());
    for (size_t i=0;i<v_in.size();i++) v_out[i]+=alpha*v_in[i];
  }*/
};


#endif // MATRIX_H
