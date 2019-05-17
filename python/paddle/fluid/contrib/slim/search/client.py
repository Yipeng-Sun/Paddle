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
"""Client for Neural Architecture Search."""
from __future__ import print_function

import time
import socket
import subprocess


class Client(object):
    """Client for Neural Architecture Search."""

    def __init__(self,
                 get_reward_command,
                 server_address,
                 buffer_size=1024,
                 verbose=False):
        """Initialize.

        Args:
            get_reward_command(list): a list command to get reward.
            server_address(tuple): a tuple of (host, port).
            buffer_size(int): buffer size.
            verbose(bool): whether to print logs.
        """
        self._get_reward_command = get_reward_command
        self._server_address = server_address
        self._buffer_size = buffer_size
        self._verbose = verbose
        self._socket_client = None

    def start(self):
        """Start client."""
        self._socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket_client.connect(self._server_address)

    def close(self):
        """Close server."""
        if self._socket_client is not None:
            self._socket_client.close()

    def run(self):
        """Run client for search."""
        while True:
            data = self._socket_client.recv(self._buffer_size).decode()
            if not data:
                break
            result = self._get_reward(data)
            data = str(result).encode()
            self._socket_client.send(data)

    def _get_reward(self, var):
        """Get reward.

        Args:
            var(str): a string that represents variable list.

        Returns:
            float: reward.
        """
        p_child = subprocess.Popen(
            self._get_reward_command(var),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        out, err = p_child.communicate()
        if self._verbose:
            print('[INFO] {} client out\n{}\nerr\n{}\n'.format(time.ctime(),
                                                               out, err))
        out = out.strip().split('\n')[-1]
        try:
            reward = float(out.split()[0])
        except ValueError as exception:
            if self._verbose:
                print('[WARN] {} exception {}'.format(time.ctime(), exception))
            reward = 0
        return reward
