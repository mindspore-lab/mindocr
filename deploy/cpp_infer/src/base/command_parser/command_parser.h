#ifndef DEPLOY_CPP_INFER_SRC_BASE_COMMAND_PARSER_COMMAND_PARSER_H_
#define DEPLOY_CPP_INFER_SRC_BASE_COMMAND_PARSER_COMMAND_PARSER_H_

#include <string>
#include <map>
#include <vector>
#include <utility>
#include "status_code/status_code.h"

// Command parser class
class CommandParser {
 public:
  CommandParser();

  CommandParser(int argc, const char **argv);

  ~CommandParser() = default;

  void AddOption(const std::string &option, const std::string &defaults = "", const std::string &message = "");

  // Parse the input arguments
  void ParseArgs(int argc, const char **argv);

  // Get the option string value from parser
  const std::string &GetStringOption(const std::string &option);

  // Get the int value by option
  int GetIntOption(const std::string &option);

  uint32_t GetUint32Option(const std::string &option);

  // Get the int value by option
  float GetFloatOption(const std::string &option);

  // Get the double option
  double GetDoubleOption(const std::string &option);

  // Get the bool option
  bool GetBoolOption(const std::string &option);

  // Get int vector
  Status GetVectorUint32Value(const std::string &option, std::vector<uint32_t> *vector);

 private :
  std::map<const std::string, std::pair<std::string, std::string>> commands_;

  // Show the usage of app, then exit
  void ShowUsage() const;

  bool IsInteger(const std::string &str) const;

  bool IsDecimal(const std::string &str) const;

  void Split(const std::string &inString, std::vector<std::string> *outVector, char delimiter);
};
#endif
