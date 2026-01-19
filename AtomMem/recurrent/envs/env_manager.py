from multiprocessing import Queue, Process
from scienceworld import ScienceWorldEnv

class SCIWorldEnvironmentWorker(Process):
    """
    每个进程控制一个环境
    Args:
        Process (_type_): _description_
    """
    def __init__(self, task_queue, shared_result_queue):
        super().__init__()
        # init_env(env_config), step(action), kill env, end worker
        self.task_queue = task_queue
        # save all the results of envs, such as _actions, _thinks, _observations, _scores, 
        self.shared_result_queue = shared_result_queue
        # environments, we hope only one environment in one environment worker
        self.env = None
        self.env_class = None
        
    def run(self):
        # override the "run" function of Process
        while True:
            # 轮询
            task = self.task_queue.get()
            if task is None:   # 终止信号，在阻塞状态下，终止信号需要手动提供
                break 
            # print(task)
            task_type, running_id, data = task
            if task_type == "init":
                # 初始化环境
                env_name, var = data["env_name"], data["var"]
                self.env: ScienceWorldEnv = ScienceWorldEnv("")
                self.env.load(env_name, var)
                init_res = {
                    "prompt": self.env.get_task_description()}
                self.shared_result_queue.put((running_id, init_res))
            
            elif task_type == "execute":
                # 执行 step
                action = data
                observation, reward, isCompleted, info = self.env.step(action)
                exe_result = {
                    "observation": observation,
                    "reward": info["score"]/100,
                    "isCompleted": isCompleted
                }
                self.shared_result_queue.put((running_id, exe_result))
            elif task_type == "clear":
                # 销毁环境
                if self.env is not None:
                    del self.env
                self.env = None
                self.env_class = None
                
class SCIWorldMultiEnvManager:
    
    def __init__(self, total_env_infos):
        self.total_env_infos = total_env_infos
        
        # 启动工作进程, 等于要部署的环境数量
        num_workers = len(total_env_infos)
        self.task_queues = [Queue() for _ in range(num_workers)]
        self.shared_result_queue = Queue()
        self.workers = []
        for i in range(num_workers):
            worker = SCIWorldEnvironmentWorker(self.task_queues[i], self.shared_result_queue)
            worker.start()
            self.workers.append(worker)
    
    def init_envs(self):
        assert len(self.total_env_infos) == len(self.workers)
        
        for running_id, env_info in enumerate(self.total_env_infos):
            self.task_queues[running_id].put(("init", running_id, env_info))
        
        initial_feedbacks = []
        random_results = self._get_results(len(self.total_env_infos))  # 获取的结果顺序随机
        for _, init_res in sorted(random_results):   # sorted by running ids
            initial_feedbacks.append(init_res)
            
        return initial_feedbacks
    
    # 支持任意指定进程的并发环境执行
    def execute_actions(self, running_ids, responses):
        assert len(responses) <= len(self.workers)
        assert len(running_ids) == len(responses)
    
        for running_id, response in zip(running_ids, responses):
            self.task_queues[running_id].put(("execute", running_id, response))

        feedbacks = []
        random_results = self._get_results(len(responses))
        for running_id, exe_res in sorted(random_results):
            feedbacks.append({"running_id":running_id, "exe_res":exe_res})
            
        return feedbacks
    
    ## auxiliary function for multiprocess envs
    def _get_results(self, expected_num):
        results = []
        while len(results) < expected_num:
            data = self.shared_result_queue.get()
            results.append(data)
        return results
    
    ## auxiliary function for multiprocess envs
    def _clear_all_envs(self):
        for task_queue in self.task_queues:
            task_queue.put(("clear", None, None))
            
    def shutdown(self):
        self._clear_all_envs()
        for q in self.task_queues:
            q.put(None)
        for worker in self.workers:
            worker.join()