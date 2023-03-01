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

#include <sstream>
#include <functional>
#include "ConfigParser.h"


namespace {
const char COMMENT_CHARATER = '#';

// Breaks the string at the separator (char) and returns a list of strings
void Split(const std::string &inString, std::vector<std::string> &outVector, const char delimiter)
{
    std::stringstream ss(inString);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        outVector.push_back(item);
    }
}
}

// Remove all spaces from the string
inline void ConfigParser::RemoveAllSpaces(std::string &str) const
{
    str.erase(std::remove_if(str.begin(), str.end(), isspace), str.end());
    return;
}

// Remove spaces from both left and right based on the string
inline void ConfigParser::Trim(std::string &str) const
{
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), std::not1(std::ptr_fun(::isspace))));
    str.erase(std::find_if(str.rbegin(), str.rend(), std::not1(std::ptr_fun(::isspace))).base(), str.end());
    return;
}
APP_ERROR ConfigParser::ParseConfig(const std::string &fileName)
{
    // Open the input file
    std::ifstream inFile(fileName);
    if (!inFile.is_open()) {
        std::cout << "cannot read setup.config file!" << std::endl;
        return APP_ERR_COMM_EXIST;
    }
    std::string line, newLine;
    int startPos, endPos, pos;
    // Cycle all the line
    while (getline(inFile, line)) {
        if (line.empty()) {
            continue;
        }
        startPos = 0;
        endPos = line.size() - 1;
        pos = line.find(COMMENT_CHARATER); // Find the position of comment
        if (pos != -1) {
            if (pos == 0) {
                continue;
            }
            endPos = pos - 1;
        }
        newLine = line.substr(startPos, (endPos - startPos) + 1); // delete comment
        pos = newLine.find('=');
        if (pos == -1) {
            continue;
        }
        std::string na = newLine.substr(0, pos);
        Trim(na); // Delete the space of the key name
        std::string value = newLine.substr(pos + 1, endPos + 1 - (pos + 1));
        Trim(value);                                   // Delete the space of value
        configData_.insert(std::make_pair(na, value)); // Insert the key-value pairs into configData_
    }
    return APP_ERR_OK;
}

// Get the string value by key name
APP_ERROR ConfigParser::GetStringValue(const std::string &name, std::string &value) const
{
    if (configData_.count(name) == 0) {
        return APP_ERR_COMM_NO_EXIST;
    }
    value = configData_.find(name)->second;
    return APP_ERR_OK;
}

// Get the int value by key name
APP_ERROR ConfigParser::GetIntValue(const std::string &name, int &value) const
{
    if (configData_.count(name) == 0) {
        return APP_ERR_COMM_NO_EXIST;
    }
    std::string str = configData_.find(name)->second;
    if (!(std::stringstream(str) >> value)) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

// Get the unsigned integer value by key name
APP_ERROR ConfigParser::GetUnsignedIntValue(const std::string &name, unsigned int &value) const
{
    if (configData_.count(name) == 0) {
        return APP_ERR_COMM_NO_EXIST;
    }
    std::string str = configData_.find(name)->second;
    if (!(std::stringstream(str) >> value)) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

// Get the bool value
APP_ERROR ConfigParser::GetBoolValue(const std::string &name, bool &value) const
{
    if (configData_.count(name) == 0) {
        return APP_ERR_COMM_NO_EXIST;
    }
    std::string str = configData_.find(name)->second;
    if (str == "true") {
        value = true;
    } else if (str == "false") {
        value = false;
    } else {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

// Get the float value
APP_ERROR ConfigParser::GetFloatValue(const std::string &name, float &value) const
{
    if (configData_.count(name) == 0) {
        return APP_ERR_COMM_NO_EXIST;
    }
    std::string str = configData_.find(name)->second;
    if (!(std::stringstream(str) >> value)) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

// Get the double value
APP_ERROR ConfigParser::GetDoubleValue(const std::string &name, double &value) const
{
    if (configData_.count(name) == 0) {
        return APP_ERR_COMM_NO_EXIST;
    }
    std::string str = configData_.find(name)->second;
    if (!(std::stringstream(str) >> value)) {
        return APP_ERR_COMM_INVALID_PARAM;
    }
    return APP_ERR_OK;
}

// Array like 1,2,4,8  split by ","
APP_ERROR ConfigParser::GetVectorUint32Value(const std::string &name, std::vector<uint32_t> &vector) const
{
    if (configData_.count(name) == 0) {
        return APP_ERR_COMM_NO_EXIST;
    }
    std::string str = configData_.find(name)->second;
    std::vector<std::string> splits;
    Split(str, splits, ',');
    uint32_t value = 0;
    std::stringstream ss;
    for (auto &it : splits) {
        if (!it.empty()) {
            std::stringstream ss(it);
            ss << it;
            ss >> value;
            vector.push_back(value);
        }
    }
    return APP_ERR_OK;
}

// new config
void ConfigParser::NewConfig(const std::string &fileName)
{
    outfile_.open(fileName, std::ios::app);
    return;
}

void ConfigParser::WriteString(const std::string &key, const std::string &value)
{
    outfile_ << key << " = " << value << std::endl;
    return;
}

void ConfigParser::WriteInt(const std::string &key, const int &value)
{
    outfile_ << key << " = " << value << std::endl;
    return;
}

void ConfigParser::WriteBool(const std::string &key, const bool &value)
{
    outfile_ << key << " = " << value << std::endl;
    return;
}

void ConfigParser::WriteFloat(const std::string &key, const float &value)
{
    outfile_ << key << " = " << value << std::endl;
    return;
}

void ConfigParser::WriteDouble(const std::string &key, const double &value)
{
    outfile_ << key << " = " << value << std::endl;
    return;
}

void ConfigParser::WriteUint32(const std::string &key, const uint32_t &value)
{
    outfile_ << key << " = " << value << std::endl;
    return;
}

void ConfigParser::SaveConfig()
{
    outfile_.close();
}