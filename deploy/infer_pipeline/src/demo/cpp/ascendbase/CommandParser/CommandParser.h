/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMAND_PARSER_H
#define COMMAND_PARSER_H

#include <string>
#include <map>
#include <vector>

// Command parser class
class CommandParser {
public:
    CommandParser();
    // Construct a new Command Parser object according to the argument
    CommandParser(int argc, const char **argv);
    ~CommandParser() {};
    // Add options into the map
    void AddOption(const std::string &option, const std::string &defaults = "", const std::string &message = "");
    // Parse the input arguments
    void ParseArgs(int argc, const char **argv);
    // Get the option string value from parser
    const std::string &GetStringOption(const std::string &option);
    // Get the int value by option
    const int GetIntOption(const std::string &option);
    const uint32_t GetUint32Option(const std::string &option);
    // Get the int value by option
    const float GetFloatOption(const std::string &option);
    // Get the double option
    const double GetDoubleOption(const std::string &option);
    // Get the bool option
    const bool GetBoolOption(const std::string &option);

private:
    std::map<std::string, std::pair<std::string, std::string>> commands_;
    // Show the usage of app, then exit
    void ShowUsage() const;
    bool IsInteger(std::string &str) const;
    bool IsDecimal(std::string &str) const;
};
#endif /* COMMANDPARSER_H */
