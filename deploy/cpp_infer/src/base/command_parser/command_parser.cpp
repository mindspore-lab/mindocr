#include <algorithm>
#include <utility>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "command_parser/command_parser.h"

namespace {

const int DEFAULT_LENGTH = 30;

const int MOD2 = 2;
}

CommandParser::CommandParser() {
  commands_["-h"] = std::make_pair("help", "show helps");
  commands_["-help"] = std::make_pair("help", "show helps");
}

void CommandParser::AddOption(
    const std::string &option,
    const std::string &defaults,
    const std::string &message) {
  commands_[option] = std::make_pair(defaults, message);
}

// Construct a new Command Parser object according to the argument
// Attention: This function may cause the program to exit directly
CommandParser::CommandParser(int argc, const char **argv) {
  ParseArgs(argc, argv);
}

void CommandParser::ParseArgs(int argc, const char **argv) {
  if (argc % MOD2 == 0) {
    ShowUsage();
  }
  for (int i = 1; i < argc; ++i) {
    std::string input(argv[i]);
    if (input == "-h" || input == "-help") {
      ShowUsage();
    }
  }
  for (int i = 1; i < argc; ++i) {
    if (i + 1 < argc && argv[i][0] == '-' && argv[i + 1][0] != '-') {
      ++i;
      continue;
    }
    ShowUsage();
  }
  for (int i = 1; i < argc; ++i) {
    if (commands_.find(argv[i]) == commands_.end()) {
      ShowUsage();
    }
    ++i;
  }
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '-') {
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        commands_[argv[i]].first = argv[i + 1];
        ++i;
      }
    }
  }
}

const std::string &CommandParser::GetStringOption(const std::string &option) {
  if (commands_.find(option) == commands_.end()) {
    std::cout << "GetStringOption fail, can not find the option "
              << option << ", make sure the option is correct!"
              << std::endl;
    ShowUsage();
  }
  return commands_[option].first;
}

// Get the int value by option

int CommandParser::GetIntOption(const std::string &option) {
  std::string str = GetStringOption(option);
  if (!IsInteger(str)) {
    std::cout << "input value "
              << str << " after" << option << " is invalid" << std::endl;
    ShowUsage();
  }
  std::stringstream ss(str);
  int value = 0;
  ss >> value;
  return value;
}

// Get the uint32 value by option
uint32_t CommandParser::GetUint32Option(const std::string &option) {
  std::string str = GetStringOption(option);
  if (!IsInteger(str)) {
    std::cout << "input value "
              << str << " after"
              << option << " is invalid" << std::endl;
    ShowUsage();
  }
  std::stringstream ss(str);
  uint32_t value = 0;
  ss >> value;
  return value;
}

// Get the int value by option
float CommandParser::GetFloatOption(const std::string &option) {
  std::string str = GetStringOption(option);
  if (!IsDecimal(str)) {
    std::cout << "input value " << str
              << " after" << option << " is invalid" << std::endl;
    ShowUsage();
  }
  std::stringstream ss(str);
  float value = 0.0;
  ss >> value;
  return value;
}

// Get the double option
double CommandParser::GetDoubleOption(const std::string &option) {
  std::string str = GetStringOption(option);
  if (!IsDecimal(str)) {
    std::cout << "input value " << str
              << " after" << option << " is invalid" << std::endl;
    ShowUsage();
  }
  std::stringstream ss(str);
  double value = 0.0;
  ss >> value;
  return value;
}

// Get the bool option
bool CommandParser::GetBoolOption(const std::string &option) {
  std::string str = GetStringOption(option);
  if (str == "true" || str == "True" || str == "TRUE") {
    return true;
  } else if (str == "false" || str == "False" || str == "FALSE") {
    return false;
  } else {
    std::cout << "GetBoolOption fail,"
              << "make sure you set the correct value true or false, but not " << str;
    ShowUsage();
    return false;
  }
}

Status CommandParser::GetVectorUint32Value
    (const std::string &option, std::vector<uint32_t> *vector) {
  std::string str = GetStringOption(option);
  std::vector<std::string> splits;
  Split(str, &splits, ',');
  uint32_t value = 0;
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

// Show the usage of app, then exit
void CommandParser::ShowUsage() const {
  std::string space(DEFAULT_LENGTH, ' ');
  std::string split(DEFAULT_LENGTH, '-');
  std::cout << std::endl << split << "help information" << split << std::endl;
  std::cout.setf(std::ios::left);
  for (auto &it : commands_) {
    if (it.first.size() >= DEFAULT_LENGTH) {
      std::cout << it.first << std::endl;
      if (it.second.first.size() >= DEFAULT_LENGTH) {
        std::cout << space << it.second.first << std::endl;
        std::cout << space << space << it.second.second << std::endl;
        continue;
      }
      std::cout << std::setw(DEFAULT_LENGTH) << it.second.first
                << std::setw(DEFAULT_LENGTH) << it.second.second
                << std::endl;
      continue;
    }
    if (it.second.first.size() >= DEFAULT_LENGTH) {
      std::cout << std::setw(DEFAULT_LENGTH)
                << it.first << std::setw(DEFAULT_LENGTH) << it.second.first
                << std::endl;
      std::cout << space << space << std::setw(DEFAULT_LENGTH)
                << it.second.second << std::endl;
      continue;
    }

    std::cout << std::setw(DEFAULT_LENGTH) << it.first
              << std::setw(DEFAULT_LENGTH) << it.second.first
              << std::setw(DEFAULT_LENGTH) << it.second.second
              << std::endl;
  }
  std::cout.setf(std::ios::right);
  std::cout << std::endl;
  exit(0);
}

bool CommandParser::IsInteger(const std::string &str) const {
  for (size_t i = 0; i < str.size(); ++i) {
    if (i == 0 && str[i] == '-') {
      continue;
    }
    if (str[i] < '0' || str[i] > '9') {
      return false;
    }
  }
  return true;
}

bool CommandParser::IsDecimal(const std::string &str) const {
  size_t dotNum = 0;
  for (size_t i = 0; i < str.size(); ++i) {
    if (i == 0 && str[i] == '-') {
      continue;
    }
    if (str[i] == '.') {
      ++dotNum;
      continue;
    }
    if (str[i] < '0' || str[i] > '9') {
      return false;
    }
  }
  if (dotNum <= 1) {
    return true;
  } else {
    return false;
  }
}

void CommandParser::Split(const std::string &inString, std::vector<std::string> *outVector, const char delimiter) {
  std::stringstream ss(inString);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    outVector->push_back(item);
  }
}
