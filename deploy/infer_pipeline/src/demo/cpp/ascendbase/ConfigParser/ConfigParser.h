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

#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "ErrorCode/ErrorCode.h"


class ConfigParser {
public:
    // Read the config file and save the useful infomation with the key-value pairs format in configData_
    APP_ERROR ParseConfig(const std::string &fileName);
    // Get the string value by key name
    APP_ERROR GetStringValue(const std::string &name, std::string &value) const;
    // Get the int value by key name
    APP_ERROR GetIntValue(const std::string &name, int &value) const;
    // Get the unsigned int value by key name
    APP_ERROR GetUnsignedIntValue(const std::string &name, unsigned int &value) const;
    // Get the bool value by key name
    APP_ERROR GetBoolValue(const std::string &name, bool &value) const;
    // Get the float value by key name
    APP_ERROR GetFloatValue(const std::string &name, float &value) const;
    // Get the double value by key name
    APP_ERROR GetDoubleValue(const std::string &name, double &value) const;
    // Get the vector by key name, split by ","
    APP_ERROR GetVectorUint32Value(const std::string &name, std::vector<uint32_t> &vector) const;

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
    std::map<std::string, std::string> configData_ = {}; // Variable to store key-value pairs
    std::ofstream outfile_ = {};

    inline void RemoveAllSpaces(std::string &str) const;
    // Remove spaces from both left and right based on the string
    inline void Trim(std::string &str) const;
};

#endif
