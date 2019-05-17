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
"""Enumerate controller."""
from itertools import product

from .controller import Controller


class EnumerateController(Controller):
    """Enumerate controller."""

    def __init__(self, range_table):
        """Initialize.

        Args:
            range_table(list): variable range table.
            reduce_rate(float): reduce rate.
            init_temperature(float): init temperature.
        """
        super(EnumerateController, self).__init__()
        self._vars = list(product(*[range(_) for _ in range_table]))
        self._total = len(self._vars)
        self._index = 0

    def check(self, reward_new, reward, iteration):
        """Check if the var should be updated using general policy.

        Args:
            reward_new(float): new reward.
            reward(float): reward.
            iteration(int): iteration.

        Returns:
            bool: a list of new variables.
        """
        return reward_new > reward

    def generate_init_var(self):
        """Generate init var.

        Returns:
            list: a list of variables.
        """
        return self._vars[0]

    def generate_new_var(self, var):
        """Generate new var.

        Args:
            var(list): a list of variables.

        Returns:
            list: a list of new variables.
        """
        self._index = (self._index + 1) % self._total
        return self._vars[self._index]
