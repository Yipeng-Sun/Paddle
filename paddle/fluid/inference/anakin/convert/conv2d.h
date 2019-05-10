// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/inference/anakin/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionT>
class Conv2dOpConverter : public AnakinOpConverter<TargetT, PrecisionT> {
 public:
  Conv2dOpConverter() = default;

  virtual void operator()(const framework::proto::OpDesc &op,
                          const framework::BlockDesc &block_desc,
                          const framework::Scope &scope,
                          bool test_mode) override;
  virtual ~Conv2dOpConverter() {}
};

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
