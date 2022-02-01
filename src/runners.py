from multiprocessing import Process

import numpy as np
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float, c_double


# idea to parallelize the training: https://arxiv.org/abs/1705.04862, https://arxiv.org/abs/1507.04296 and https://arxiv.org/abs/1803.02811
# did not finish -> work in progress / future work
class EnvironmentRunner(Process):
    def __init__(self, runner_id, environments, variables, queue, barrier):
        super(EnvironmentRunner, self).__init__()
        self.runner_id = runner_id
        self.environments = environments
        self.variables = variables
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super(EnvironmentRunner, self).run()
        self._run()

    def _run(self):
        count = 0
        while True:
            instruction = self.queue.get()
            if instruction is None:
                break
            for i, (environment, action) in enumerate(zip(self.environments, self.variables[-1])):
                new_s, reward, episode_over, _ = environment.step(action)
                if episode_over:
                    self.variables[0][i] = environment.reset()
                else:
                    self.variables[0][i] = new_s
                self.variables[1][i] = reward
                self.variables[2][i] = episode_over
            count += 1
            self.barrier.put(True)


class Runners(object):
    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8: c_uint}

    def __init__(self, environments, variables):
        self.variables = [self._get_shared(var) for var in variables]
        self.num_workers = len(environments)
        self.queues = [Queue() for _ in range(self.num_workers)]
        self.barrier = Queue()

        self.runners = [EnvironmentRunner(i, environments, vars, self.queues[i], self.barrier) for
                        i, (environments, vars) in enumerate(zip(np.split(environments, self.num_workers),
                                                                 zip(*[np.split(var, self.num_workers) for var in
                                                                       self.variables])))]

    def _get_shared(self, array):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :return: the RawArray backed numpy array
        """

        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def start(self):
        for r in self.runners:
            r.start()

    def stop(self):
        for queue in self.queues:
            queue.put(None)

    def get_shared_variables(self):
        return self.variables

    def perform_step(self):
        for queue in self.queues:
            queue.put(True)

    def wait_updated(self):
        for _ in range(self.workers):
            self.barrier.get()
