#ifndef DEPLOY_CPP_INFER_SRC_PARALLEL_FRAMEWORK_MODULE_FACTORY_H_
#define DEPLOY_CPP_INFER_SRC_PARALLEL_FRAMEWORK_MODULE_FACTORY_H_


#include <string>
#include <map>
#include <functional>

#define MODULE_REGIST(class_name)                                                         \
    namespace AscendBaseModule {                                                          \
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
    }  // namespace AscendBaseModule

namespace AscendBaseModule {
using Constructor = std::function<void *()>;

class ModuleFactory {
 public:
    static void RegisterModule(std::string className, Constructor constructor) {
        Constructors()[className] = constructor;
    }

    static void *MakeModule(const std::string &className) {
        auto itr = Constructors().find(className);
        if (itr == Constructors().end()) {
            return nullptr;
        }
        return ((Constructor)itr->second)();
    }

 private:
    inline static std::map<std::string, Constructor> &Constructors() {
        static std::map<std::string, Constructor> instance;
        return instance;
    }
};
}  // namespace AscendBaseModule
#endif
