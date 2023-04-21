#include <sstream>
#include <functional>
#include <utility>
#include "config_parser/config_parser.h"

const char COMMENT_CHARACTER = '#';

// Breaks the string at the separator (char) and returns a list of strings
void Split(const std::string &inString, std::vector<std::string> *outVector, const char delimiter) {
  std::stringstream ss(inString);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    outVector->push_back(item);
  }
}

// Remove all spaces from the string
inline void ConfigParser::RemoveAllSpaces(std::string *str) {
  str->erase(std::remove_if(str->begin(), str->end(), isspace), str->end());
}

// Remove spaces from both left and right based on the string
inline void ConfigParser::Trim(std::string *str) {
  str->erase(str->begin(), std::find_if(str->begin(), str->end(), std::not1(std::ptr_fun(::isspace))));
  str->erase(std::find_if(str->rbegin(), str->rend(), std::not1(std::ptr_fun(::isspace))).base(), str->end());
}

Status ConfigParser::ParseConfig(const std::string &fileName) {
  // Open the input file
  std::ifstream inFile(fileName);
  if (!inFile.is_open()) {
    std::cout << "cannot read setup.config file!" << std::endl;
    return Status::COMM_NO_EXIST;
  }
  std::string line;
  std::string newLine;
  int startPos;
  int endPos;
  int pos;
  // Cycle all the line
  while (getline(inFile, line)) {
    if (line.empty()) {
      continue;
    }
    startPos = 0;
    endPos = line.size() - 1;
    pos = line.find(COMMENT_CHARACTER);  // Find the position of comment
    if (pos != -1) {
      if (pos == 0) {
        continue;
      }
      endPos = pos - 1;
    }
    newLine = line.substr(startPos, (endPos - startPos) + 1);  // delete comment
    pos = newLine.find('=');
    if (pos == -1) {
      continue;
    }
    std::string na = newLine.substr(0, pos);
    Trim(&na);  // Delete the space of the key name
    std::string value = newLine.substr(pos + 1, endPos + 1 - (pos + 1));
    Trim(&value);
    configData_.insert(std::make_pair(na, value));  // Insert the key-value pairs into configData_
  }
  return Status::OK;
}

// Get the string value by key name
Status ConfigParser::GetStringValue(const std::string &name, std::string *value) const {
  if (configData_.count(name) == 0) {
    return Status::COMM_NO_EXIST;
  }
  *value = configData_.find(name)->second;
  return Status::OK;
}

// Get the int value by key name
Status ConfigParser::GetIntValue(const std::string &name, int *value) const {
  if (configData_.count(name) == 0) {
    return Status::COMM_NO_EXIST;
  }
  auto iter = configData_.find(name);
  if (iter == configData_.end()) {
    return Status::COMM_NO_EXIST;
  }
  try {
    *value = std::stoi(iter->second);
  } catch (const std::exception &e) {
    return Status::COMM_INVALID_PARAM;
  }
  return Status::OK;
}

// Get the unsigned integer value by key name
Status ConfigParser::GetUnsignedIntValue(const std::string &name, unsigned int *value) const {
  if (configData_.count(name) == 0) {
    return Status::COMM_NO_EXIST;
  }
  std::string str = configData_.find(name)->second;
  if (!(std::stringstream(str) >> *value)) {
    return Status::COMM_INVALID_PARAM;
  }
  return Status::OK;
}

// Get the bool value
Status ConfigParser::GetBoolValue(const std::string &name, bool *value) const {
  if (configData_.count(name) == 0) {
    return Status::COMM_NO_EXIST;
  }
  std::string str = configData_.find(name)->second;
  if (str == "true") {
    *value = true;
  } else if (str == "false") {
    *value = false;
  } else {
    return Status::COMM_INVALID_PARAM;
  }
  return Status::OK;
}

// Get the float value
Status ConfigParser::GetFloatValue(const std::string &name, float *value) const {
  if (configData_.count(name) == 0) {
    return Status::COMM_NO_EXIST;
  }
  std::string str = configData_.find(name)->second;
  if (!(std::stringstream(str) >> *value)) {
    return Status::COMM_INVALID_PARAM;
  }
  return Status::OK;
}

// Get the double value
Status ConfigParser::GetDoubleValue(const std::string &name, double *value) const {
  if (configData_.count(name) == 0) {
    return Status::COMM_NO_EXIST;
  }
  std::string str = configData_.find(name)->second;
  if (!(std::stringstream(str) >> *value)) {
    return Status::COMM_INVALID_PARAM;
  }
  return Status::OK;
}

// Array like 1,2,4,8  split by ","
Status ConfigParser::GetVectorUint32Value(const std::string &name, std::vector<uint32_t> *vector) const {
  if (configData_.count(name) == 0) {
    return Status::COMM_NO_EXIST;
  }
  std::string str = configData_.find(name)->second;
  std::vector<std::string> splits;
  Split(str, &splits, ',');
  uint32_t value = 0;
  std::stringstream ss;
  for (auto &it : splits) {
    if (!it.empty()) {
      std::stringstream ss(it);
      ss << it;
      ss >> value;
      vector->push_back(value);
    }
  }
  return Status::OK;
}

// new config
void ConfigParser::NewConfig(const std::string &fileName) {
  outfile_.open(fileName, std::ios::app);
}

void ConfigParser::WriteString(const std::string &key, const std::string &value) {
  outfile_ << key << " = " << value << std::endl;
}

void ConfigParser::WriteInt(const std::string &key, const int &value) {
  outfile_ << key << " = " << value << std::endl;
}

void ConfigParser::WriteBool(const std::string &key, const bool &value) {
  outfile_ << key << " = " << value << std::endl;
}

void ConfigParser::WriteFloat(const std::string &key, const float &value) {
  outfile_ << key << " = " << value << std::endl;
}

void ConfigParser::WriteDouble(const std::string &key, const double &value) {
  outfile_ << key << " = " << value << std::endl;
}

void ConfigParser::WriteUint32(const std::string &key, const uint32_t &value) {
  outfile_ << key << " = " << value << std::endl;
}

void ConfigParser::SaveConfig() {
  outfile_.close();
}
