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
"""Server for Neural Architecture Search."""
from __future__ import print_function

import time
import socket
from multiprocessing import Process, Queue


class Server(object):
    """Server for Neural Architecture Search."""

    def __init__(self,
                 controller,
                 address,
                 num_clients,
                 buffer_size=1024,
                 verbose=False):
        """Initialize.

        Args:
            controller(Controller object): controller.
            address(tuple): a tuple of (host, port).
            num_clients(int): number of clients.
            buffer_size(int): buffer size.
            verbose(bool): whether to print logs.
        """
        self._controller = controller
        self._address = address
        self._num_clients = num_clients
        self._buffer_size = buffer_size
        self._verbose = verbose
        self._socket_server = None

    def start(self):
        """Start server."""
        self._socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket_server.bind(self._address)
        self._socket_server.listen(self._num_clients)

    def close(self):
        """Close server."""
        if self._socket_server is not None:
            self._socket_server.close()

    def run(self, max_iterations, init_var, init_reward):
        """Run server for search.

        Args:
            max_iterations(int): max iterations.
            init_var(list): init var.
            init_reward(float): init reward.

        Returns:
            tuple: a tuple of (var, reward)
                where reward is the maximum during searching.
        """
        list_socket_clients = self._get_clients()
        var = init_var
        reward = init_reward
        var_max = var
        reward_max = reward
        if self._verbose:
            print('[INFO] {} iter 0 reward {} var {}'.format(time.ctime(),
                                                             reward, var))
        task_queue = Queue()
        done_queue = Queue()
        for client_index in range(self._num_clients):
            var_new = self._controller.generate_new_var(var)
            task_queue.put((client_index, var_new))
            Process(
                target=self._worker,
                args=(task_queue, done_queue, list_socket_clients)).start()
        for iteration in range(1, max_iterations + 1):
            client_index, var_new, reward_new = done_queue.get()
            if self._verbose:
                print('[INFO] {} client {} reward_new {} var_new {}'.format(
                    time.ctime(), client_index, reward_new, var_new))
            if self._controller.check(reward_new, reward, iteration):
                reward = reward_new
                var = var_new[:]
            if reward_new > reward_max:
                reward_max = reward_new
                var_max = var
            if self._verbose:
                print('[INFO] {} iter {} reward {} var {}'.format(time.ctime(
                ), iteration, reward, var))
            var_new = self._controller.generate_new_var(var)
            task_queue.put((client_index, var_new))
        for client_index in range(self._num_clients):
            task_queue.put('STOP')
            list_socket_clients[client_index].close()
        return (var_max, reward_max)

    def _get_clients(self):
        """Get clients.

        Returns:
            list: a list of client sockets.
        """
        list_socket_clients = list()
        for client_index in range(self._num_clients):
            socket_client, address = self._socket_server.accept()
            if self._verbose:
                print('[INFO] client {} {}'.format(client_index, address))
            list_socket_clients.append(socket_client)
        return list_socket_clients

    def _get_reward_from_socket(self, socket_client, data):
        """Get reward from socket.

        Args:
            socket_client(socket): socket of the client.
            data(str): data string.

        Returns:
            float: reward.
        """
        data = data.encode()
        socket_client.send(data)
        data = socket_client.recv(self._buffer_size).decode()
        out = data.strip().split('\n')[-1]
        try:
            reward = float(out.split()[0])
        except ValueError as err:
            print('[WARN] {} out {} err {}'.format(time.ctime(), out, err))
            reward = 0
        return reward

    def _worker(self, input_queue, output_queue, list_socket_clients):
        """Worker for one process.

        Args:
            input_queue(Queue object): input queue.
            output_queue(Queue object): output queue.
            list_socket_clients(list): a list of socket_clients.
        """
        for client_index, var in iter(input_queue.get, 'STOP'):
            result = self._get_reward_from_socket(
                list_socket_clients[client_index], str(var).replace(' ', ''))
            output_queue.put((client_index, var, result))
