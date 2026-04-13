// Copyright 2016 Google Inc.
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
// limitations under the License.!

#ifndef ABSL_CONTAINER_FLAT_HASH_SET_
#define ABSL_CONTAINER_FLAT_HASH_SET_

#include "third_party/ankerl/unordered_dense.h"

namespace absl {

template <typename T,
          typename Hash = ankerl::unordered_dense::hash<T>,
          typename Eq = std::equal_to<T>>
using flat_hash_set = ankerl::unordered_dense::set<T, Hash, Eq>;

}

#endif  // ABSL_CONTAINER_FLAT_HASH_SET_
