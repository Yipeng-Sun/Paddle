/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/lodtensor_printer.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace platform {

template <typename T>
void print_lod_tensor(const std::string& var_name,
                      const framework::LoDTensor& lod_tensor,
                      const std::string& print_info) {
  auto inspect = lod_tensor.data<T>();
  auto element_num = lod_tensor.numel();

  std::ostringstream sstream;
  sstream << print_info << "\t";
  sstream << var_name << "\t";
  sstream << inspect[0];
  for (int j = 1; j < element_num; ++j) {
    sstream << " " << inspect[j];
  }

  std::cout << sstream.str() << std::endl;
}

void PrintVar(framework::Scope* scope, const std::string& var_name,
              const std::string& print_info) {
  framework::Variable* var = scope->FindVar(var_name);
  if (var == nullptr) {
    VLOG(1) << "Variable Name " << var_name << " does not exist in your scope";
    return;
  }
  framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();
  if (tensor == nullptr) {
    VLOG(1) << "tensor of variable " << var_name
            << " does not exist in your scope";
    return;
  }

  framework::LoDTensor printed_tensor;
  printed_tensor.set_lod(tensor->lod());
  printed_tensor.Resize(tensor->dims());
  if (platform::is_cpu_place(tensor->place())) {
    printed_tensor.ShareDataWith(*tensor);
  } else {
    platform::CPUPlace place;
    framework::TensorCopy(*tensor, place, &printed_tensor);
  }

#define PrintLoDTensorCallback(cpp_type, proto_type)                    \
  do {                                                                  \
    if (tensor->type() == proto_type) {                                 \
      print_lod_tensor<cpp_type>(var_name, printed_tensor, print_info); \
      return;                                                           \
    }                                                                   \
  } while (0)

  _ForEachDataType_(PrintLoDTensorCallback);
  VLOG(1) << "PrintVar: unrecognized data type:" << printed_tensor.type();
}

}  // end namespace platform
}  // end namespace paddle
