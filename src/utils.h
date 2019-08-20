#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <fstream>

namespace SLUTILS {

void SplitToken(const std::string& str,std::vector<std::string>& tokens,const std::string& delimiters)
{
  auto lastPos = str.find_first_not_of(delimiters, 0); // Skip delimiters at beginning.
  auto pos     = str.find_first_of(delimiters, lastPos); // Find first "non-delimiter".

  while (std::string::npos != pos || std::string::npos != lastPos)  {
      tokens.push_back(str.substr(lastPos, pos - lastPos)); // Found a token, add it to the vector.
      lastPos = str.find_first_not_of(delimiters, pos); // Skip delimiters.  Note the "not_of"
      pos = str.find_first_of(delimiters, lastPos); // Find next "non-delimiter"
  }
}

int ReadCSVData(const std::string &fname,const std::string &delim,std::vector<std::vector<std::string>> &data)
{
  using std::string;
  using std::vector;

  std::ifstream is(fname);
  if (is.good()) {
    string line;
    while (std::getline(is,line)) {
      vector <string> tokens;
      SplitToken(line,tokens,delim);
      data.push_back(tokens);
    }
    return 0;
  } else return 1;
}

};
#endif // UTILS_H
