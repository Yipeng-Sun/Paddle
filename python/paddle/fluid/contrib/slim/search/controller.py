#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Controller for Neural Architecture Search.
"""


class Controller(object):
    """Controller for Neural Architecture Search.
    """

    def __init__(self, *args, **kwargs):
        pass

    def check(self, reward_new, reward, iteration):
        """Check if the var should be updated.
        """
        raise NotImplementedError('Abstract method.')

    def generate_init_var(self):
        """Generate init var.
        """
        raise NotImplementedError('Abstract method.')

    def generate_new_var(self, var):
        """Generate new var.
        """
        raise NotImplementedError('Abstract method.')
