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

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "CommandParser.h"

namespace {
const int DEFAULT_LENGTH = 30; // The length of delimiter for help information
const int MOD2 = 2; // Constant to make sure the parameters after ./main is pairs
}

CommandParser::CommandParser()
{
    commands_["-h"] = std::make_pair("help", "show helps");
    commands_["-help"] = std::make_pair("help", "show helps");
}


// Add options into the map
void CommandParser::AddOption(const std::string &option, const std::string &defaults, const std::string &message)
{
    commands_[option] = std::make_pair(defaults, message);
}


// Construct a new Command Parser object according to the argument
// Attention: This function may cause the program to exit directly
CommandParser::CommandParser(int argc, const char **argv)
{
    ParseArgs(argc, argv);
}

// Attention: This function will cause the program to exit directly when calling ShowUsage()
void CommandParser::ParseArgs(int argc, const char **argv)
{
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

// Get the option string value from parser
// Attention: This function will cause the program to exit directly when calling ShowUsage()
const std::string &CommandParser::GetStringOption(const std::string &option)
{
    if (commands_.find(option) == commands_.end()) {
        std::cout << "GetStringOption fail, can not find the option " << option << ", make sure the option is correct!"
                  << std::endl;
        ShowUsage();
    }
    return commands_[option].first;
}

// Get the int value by option
// Attention: This function will cause the program to exit directly when calling ShowUsage()
const int CommandParser::GetIntOption(const std::string &option)
{
    std::string str = GetStringOption(option);
    if (!IsInteger(str)) {
        std::cout << "input value " << str << " after" << option << " is invalid" << std::endl;
        ShowUsage();
    }
    std::stringstream ss(str);
    int value = 0;
    ss >> value;
    return value;
}

// Get the uint32 value by option
// Attention: This function will cause the program to exit directly when calling ShowUsage()
const uint32_t CommandParser::GetUint32Option(const std::string &option)
{
    std::string str = GetStringOption(option);
    if (!IsInteger(str)) {
        std::cout << "input value " << str << " after" << option << " is invalid" << std::endl;
        ShowUsage();
    }
    std::stringstream ss(str);
    uint32_t value = 0;
    ss >> value;
    return value;
}

// Get the int value by option
// Attention: This function will cause the program to exit directly when calling ShowUsage()
const float CommandParser::GetFloatOption(const std::string &option)
{
    std::string str = GetStringOption(option);
    if (!IsDecimal(str)) {
        std::cout << "input value " << str << " after" << option << " is invalid" << std::endl;
        ShowUsage();
    }
    std::stringstream ss(str);
    float value = 0.0;
    ss >> value;
    return value;
}

// Get the double option
// Attention: This function will cause the program to exit directly when calling ShowUsage()
const double CommandParser::GetDoubleOption(const std::string &option)
{
    std::string str = GetStringOption(option);
    if (!IsDecimal(str)) {
        std::cout << "input value " << str << " after" << option << " is invalid" << std::endl;
        ShowUsage();
    }
    std::stringstream ss(str);
    double value = 0.0;
    ss >> value;
    return value;
}

// Get the bool option
// Attention: This function will cause the program to exit directly when calling ShowUsage()
const bool CommandParser::GetBoolOption(const std::string &option)
{
    std::string str = GetStringOption(option);
    if (str == "true" || str == "True" || str == "TRUE") {
        return true;
    } else if (str == "false" || str == "False" || str == "FALSE") {
        return false;
    } else {
        std::cout << "GetBoolOption fail, make sure you set the correct value true or false, but not " << str;
        ShowUsage();
        return false;
    }
}

// Show the usage of app, then exit
// Attention: This function will cause the program to exit directly after printing usage
void CommandParser::ShowUsage() const
{
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
            std::cout << std::setw(DEFAULT_LENGTH) << it.second.first << std::setw(DEFAULT_LENGTH) << it.second.second
                      << std::endl;
            continue;
        }
        if (it.second.first.size() >= DEFAULT_LENGTH) {
            std::cout << std::setw(DEFAULT_LENGTH) << it.first << std::setw(DEFAULT_LENGTH) << it.second.first
                      << std::endl;
            std::cout << space << space << std::setw(DEFAULT_LENGTH) << it.second.second << std::endl;
            continue;
        }
        std::cout << std::setw(DEFAULT_LENGTH) << it.first << std::setw(DEFAULT_LENGTH) << it.second.first
                  << std::setw(DEFAULT_LENGTH) << it.second.second << std::endl;
    }
    std::cout.setf(std::ios::right);
    std::cout << std::endl;
    exit(0);
}

bool CommandParser::IsInteger(std::string &str) const
{
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

bool CommandParser::IsDecimal(std::string &str) const
{
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