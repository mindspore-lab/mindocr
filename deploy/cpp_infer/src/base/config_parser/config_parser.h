#ifndef DEPLOY_CPP_INFER_SRC_BASE_CONFIG_PARSER_CONFIG_PARSER_H_
#define DEPLOY_CPP_INFER_SRC_BASE_CONFIG_PARSER_CONFIG_PARSER_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "status_code/status_code.h"

class ConfigParser {
 public:
  // Read the config file and save the useful infomation with the key-value pairs format in configData_
  Status ParseConfig(const std::string &fileName);

  // Get the string value by key name
  Status GetStringValue(const std::string &name, std::string *value) const;

  // Get the int value by key name
  Status GetIntValue(const std::string &name, int *value) const;

  // Get the unsigned int value by key name
  Status GetUnsignedIntValue(const std::string &name, unsigned int *value) const;

  // Get the bool value by key name
  Status GetBoolValue(const std::string &name, bool *value) const;

  // Get the float value by key name
  Status GetFloatValue(const std::string &name, float *value) const;

  // Get the double value by key name
  Status GetDoubleValue(const std::string &name, double *value) const;

  // Get the vector by key name, split by ","
  Status GetVectorUint32Value(const std::string &name, std::vector<uint32_t> *vector) const;

  void NewConfig(const std::string &fileName);

  // Write the values into new config file
  void WriteString(const std::string &key, const std::string &value);

  void WriteInt(const std::string &key, const int &value);

  void WriteBool(const std::string &key, const bool &value);

  void WriteFloat(const std::string &key, const float &value);

  void WriteDouble(const std::string &key, const double &value);

  void WriteUint32(const std::string &key, const uint32_t &value);

  void SaveConfig();

 private:
  std::map<std::string, std::string> configData_ = {};
  std::ofstream outfile_ = {};

  inline void RemoveAllSpaces(std::string *str);

  // Remove spaces from both left and right based on the string
  inline void Trim(std::string *str);
};
#endif
