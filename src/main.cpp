#include "ols.h"
#include "utils.h"
#include "nn/nn.h"

#include <iostream>
#include <cmath>
#include <iomanip>

double PredictTimeSeriesOLS(int order,const std::vector<double>&data,std::vector <double>&pred)
{
  if (static_cast<int>(data.size())>order) {
    double esum=0.;
    std::vector<double> x(order+1);
    OLS<double>ols(order+1);
    pred.clear();
    pred.resize(data.size());
    for (size_t i=order;i<data.size();i++) {
      for (int j=0;j<order;j++) x[j]=data[i-(j+1)];
      x[order]=1.0; // constant term
      pred[i]=ols.Predict(x);
      esum+=fabs((data[i]-pred[i]));
      ols.Update(data[i]);
    }
    double e=(esum/(static_cast<double>(data.size()-order)));
    return e;
  } else return 0;
}

double PredictTimeSeriesMLP(int order,const std::vector<double>&data,std::vector <double>&pred)
{
  if (static_cast<int>(data.size())>order) {
    const int nhidden=8;
    double esum=0.;

    NN_MLP nn(order,OPT::TYPE::ADAM);
    nn.AddLayer(nhidden,TF::TYPE::RELU);
    nn.AddLayer(1,TF::TYPE::LINEAR);
    std::vector<double> x(order);
    pred.clear();
    pred.resize(data.size());
    for (size_t i=order;i<data.size();i++) {
      for (int j=0;j<order;j++) x[j]=data[i-(j+1)];
      nn.Predict(x);
      pred[i]=nn.Output()[0];

      esum+=fabs((data[i]-pred[i]));
      nn.Update({data[i]});
    }
    double e=(esum/(static_cast<double>(data.size()-order)));
    return e;
  } else return 0;
}

// Predicts Up/Down Movement via MLP-Classifier
double PredictTimeSeriesMovement(int order,const std::vector<double>&data)
{
  if (static_cast<int>(data.size())>order) {
    const int nhidden=32;

    NN_MLP nn(order,OPT::TYPE::ADAM);
    nn.AddLayer(nhidden,TF::TYPE::RELU);
    nn.AddLayer(2,TF::TYPE::LINEAR); // higher, lower
    nn.SetSoftMax(true);

    std::vector<double> x(order);
    int64_t guess_right=0,guess_total=0;
    for (size_t i=order;i<data.size();i++) {
      for (int j=0;j<order;j++) x[j]=data[i-(j+1)];
      nn.Predict(x);

      double p0=nn.Output()[0];
      double p1=nn.Output()[1];
      int plabel=p0>0?1:0;

      int label=data[i]>0?1:0;

      if (plabel==label) guess_right++;
      guess_total++;

      if (label) nn.Update({1,0}); // goes up
      else nn.Update({0,1}); // goes down
      //nn.Update({data[i]});
    }
    return guess_right/static_cast<double>(guess_total);
  } else return 0;

}

int main()
{
    std::cout << "Stock Prediction Examples v0.1 (c) 2018,2019 Sebastian Lehmann" << std::endl;

    std::string fname("SP500w.csv");
    std::vector<std::vector<std::string>> data;
    std::cout << "Reading data from '" << fname << "': ";
    if (!SLUTILS::ReadCSVData(fname,",",data)) {
      std::cout << "ok\n";

      std::vector <double> returns;
      int num_error=0;
      double last_price=std::stod(data[1][5]);
      for (size_t i=2;i<data.size();i++) {
         double price=std::stod(data[i][5]);
         if (price>0.0) {
            //returns.push_back(price);
            returns.push_back(log(price/last_price));
            last_price=price;
         } else num_error++;
      }
      std::cout << returns.size() << " timepoints, " << num_error << " invalid points\n\n";
      std::vector <double>pred;

      std::cout << "Sequential log-return prediction\n";
      std::cout << "--------------------------------\n";

      std::cout << "Null model (last timepoint):\n";
      double esum=0;
      for (size_t i=1;i<returns.size();i++) {
        esum+=fabs((returns[i]-returns[i-1]));
      }
      double e=(esum/(static_cast<double>(returns.size()-1)));
      std::cout << "NULL:          " << std::setprecision(4) << e << "\n";

      std::cout << "OLS with cholesky decomposition:\n";
      for (int k=0;k<=3;k++) {
        int order=1<<k;
        double rmse=PredictTimeSeriesOLS(order,returns,pred);
        std::cout << "MAE OLS (o" << order << "): " << std::setprecision(4) << rmse << "\n";
      }
      std::cout << '\n';
      std::cout << "MLP with adam optimizer:\n";
      for (int k=0;k<=3;k++) {
        int order=1<<k;
        double rmse=PredictTimeSeriesMLP(order,returns,pred);
        std::cout << "MAE MLP (o" << order << "): " << std::setprecision(4) << rmse << "\n";
      }
      std::cout << '\n';
      std::cout << "Sequential Movement (Up/Down) prediction\n";
      std::cout << "----------------------------------------\n";
      std::cout << "Null model (last movement):\n";
      int64_t guess_right=0,guess_total=0;
      for (size_t i=1;i<returns.size();i++) {
        int plabel=returns[i-1]>0?1:0;
        int label=returns[i]>0?1:0;
        if (plabel==label) guess_right++;
        guess_total++;
      }
      double p=guess_right/static_cast<double>(guess_total);
      std::cout << "Estimated probability of guessing right: " << p << '\n';

      for (int k=0;k<=3;k++) {
        int order=1<<k;
        p=PredictTimeSeriesMovement(order,returns);
        std::cout << "MLP-Classifier (o" << order << "): " << p << '\n';
      }
    } else std::cout << "error!\n";
    return 0;
}
