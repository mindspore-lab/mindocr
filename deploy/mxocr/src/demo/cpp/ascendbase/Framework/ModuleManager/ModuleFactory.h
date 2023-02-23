/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INC_MODULE_FACTORY_H
#define INC_MODULE_FACTORY_H

#include <string>
#include <map>
#include <functional>

#define MODULE_REGIST(class_name)                                                         \
    namespace ascendBaseModule {                                                          \
    class class_name##Helper {                                                            \
    public:                                                                               \
        class_name##Helper()                                                              \
        {                                                                                 \
            ModuleFactory::RegisterModule(#class_name, class_name##Helper::CreatObjFunc); \
        }                                                                                 \
        static void *CreatObjFunc()                                                       \
        {                                                                                 \
            return new class_name;                                                        \
        }                                                                                 \
    };                                                                                    \
    static class_name##Helper class_name##helper;                                         \
    static std::string MT_##class_name = #class_name;                                     \
    }

namespace ascendBaseModule {
using Constructor = std::function<void *()>;

class ModuleFactory {
public:
    static void RegisterModule(std::string className, Constructor constructor)
    {
        Constructors()[className] = constructor;
    }

    static void *MakeModule(const std::string &className)
    {
        auto itr = Constructors().find(className);
        if (itr == Constructors().end()) {
            return nullptr;
        }
        return ((Constructor)itr->second)();
    }

private:
    inline static std::map<std::string, Constructor> &Constructors()
    {
        static std::map<std::string, Constructor> instance;
        return instance;
    }
};
}

#endif