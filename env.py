import numpy as np
from copy import deepcopy

class ENV:

    """
    环境包括：
        边缘计算系统 ECS：
            各边缘服务器的函数热度排名、各边缘服务器的所有类任务的总量动态变化
        函数实例部署状态 FunctionDeployStats：
            由算法动态调整
        任务卸载算法 MaxBenefit：
            收益最大
    """

    def __init__(self, function_type_num=None, edge_server_num=None, **params):
        self.params = deepcopy(params)
        self.ecs = ECS(function_type_num, edge_server_num, **params['sys_hyper_params'])
        self.ecs.gener_system()
        self.function_deploy_stats = FunctionDeployStats(self.ecs.sys_info)

        self.resource_usage_tasks_max_num = ResourceUsageTasksMaxNum(self.ecs.sys_info)
        self.simulate_function_deploy_stats = FunctionDeployStats(self.ecs.sys_info)

        self.cloud_execution = CloudExecution(self.ecs.sys_info, **params['off_hyper_params'])

        # 任务卸载算法
        self.max_benefit = MaxBenefit(self.ecs.sys_info, **params['off_hyper_params'])

        # state 为当前 slot 各边缘服务器的收到的 tasks 统计
        # state[0] 为任务量统计信息
        self.state = np.zeros(shape=(3, edge_server_num, function_type_num), dtype=int)
        self.action = None
        self.reward = None
        self.terminate = False

    def reset(self):
        """
        重置整个环境
            重置 ENV、函数实例部署状态、任务
        Returns:
            self.state
        """
        self.ecs.gener_tasks()
        self.function_deploy_stats.reset()

        self.simulate_function_deploy_stats.reset()

        self.state[0] = self.ecs.calculate_tasks_statistics('heat')
        self.state[-1] = self.function_deploy_stats.function_instance_nums
        self.action = None
        self.reward = None
        self.terminate = False

        return self.state

    def step(self, action: np.ndarray, action_type='resource_proportion', update_ecs=True):
        """
        由函数实例在边缘服务器部署的数量调整动作对应更新各类函数实例在边缘服务器的部署状态

        Args:
            action (np.ndarray): 函数实例在边缘服务器部署的数量调整动作
            action_type (str): 声明动作类型 'resource_proportion' or 'alter_nums'
            update_ecs (bool): 边缘计算系统中函数排名和任务量是否更新

        Returns:
            self.state, self.reward, self.terminate
        """
        # 得到函数实例部署调整动作
        self.action = action

        # 更新 state 中的函数实例部署状态
        self.function_deploy_stats.update(action, action_type)

        # 动作是调整个数时，需要判断是否越界
        if action_type == 'alter_nums' and not self.is_valid():
            self.terminate = True
            self.function_deploy_stats.reset()

        # todo
        # 任务卸载，计算奖励
        offloading_schedule_info = self.max_benefit.offloading(self.ecs.tasks,
                                                               self.function_deploy_stats.function_instance_stats)
        offloading_schedule_res = self.max_benefit.result(offloading_schedule_info,
                                                          self.function_deploy_stats.function_instance_stats)
        # 强化学习是 max reward，与目标相反，返回负值
        self.reward = -offloading_schedule_res['tasks_delay'] / offloading_schedule_res['tasks_num']

        # heuristic_action = self.resource_usage_tasks_max_num.make_action(self.state, 'resource_proportion')
        # self.simulate_function_deploy_stats.update(heuristic_action, 'resource_proportion')
        # simulate_function_instance_stats = self.simulate_function_deploy_stats.function_instance_stats
        # simulate_offloading_schedule_info = self.max_benefit.offloading(self.ecs.tasks,
        #                                                                 simulate_function_instance_stats)
        # simulate_offloading_schedule_res = self.max_benefit.result(simulate_offloading_schedule_info,
        #                                                            simulate_function_instance_stats)
        # # 强化学习是 max reward，与目标相反，返回负值
        # x = simulate_offloading_schedule_res['tasks_delay'] / simulate_offloading_schedule_res['tasks_num']
        # y = offloading_schedule_res['tasks_delay'] / offloading_schedule_res['tasks_num']
        # self.reward = (x - y) / x

        c_offloading_schedule_info = self.cloud_execution.offloading(self.ecs.tasks,
                                                                     self.function_deploy_stats.function_instance_stats)
        c_offloading_schedule_res = self.cloud_execution.result(c_offloading_schedule_info,
                                                                self.function_deploy_stats.function_instance_stats)
        # 强化学习是 max reward，与目标相反，返回负值
        # x = c_offloading_schedule_res['tasks_delay'] / c_offloading_schedule_res['tasks_num']
        # y = offloading_schedule_res['tasks_delay'] / offloading_schedule_res['tasks_num']
        # self.reward = (x - y) / y

        # 环境动态变化
        if update_ecs and np.random.random() < self.params['update_function_rank_proportion']:
            self.ecs.update_edge_server_function_rank(max(1, int(self.ecs.sys_info['edge_server_num'] *
                                                                 self.params['update_function_rank_probability'])))
        # 产生新的任务
        self.ecs.gener_tasks()

        # 更新下一个 state
        self.state[0] = self.ecs.calculate_tasks_statistics('heat')
        self.state[-1] = self.function_deploy_stats.function_instance_nums

        return self.state, self.reward, self.terminate

    def is_valid(self):
        """
        若动作是函数实例调整个数，调用此函数判断动作是否合法（执行动作后是否资源越界）

        Returns:
            bool
        """
        for i in range(self.ecs.sys_info['edge_server_num']):
            edge_server_id = self.ecs.sys_info['edge_server_id_set'][i]
            comp_capa = self.ecs.sys_info['edge_server_info'][edge_server_id]['comp_capa']

            for j in range(self.ecs.sys_info['function_type_num']):
                function_id = self.ecs.sys_info['function_id_set'][j]
                comp_resource = self.ecs.sys_info['function_info'][function_id]['comp_resource']
                num = self.function_deploy_stats.function_instance_nums[i][j]

                comp_capa -= (comp_resource * num)
                # edge_server_id 是否存在资源越界
                if comp_capa < 0:
                    return False

        return True

    def plt_env(self, option='tasks', time_slot=None, save_dir='', is_show=True):
        """
        绘制环境 env 的信息

        Args:
            option (str): 'tasks', 'network', 'deploy'
            time_slot (int):
            save_dir (str): 保存路径
            is_show (bool)
        """
        if time_slot is None and option != 'network':
            raise TypeError("'time_slot' unspecified.")

        if option == 'tasks':
            self.ecs.plt_tasks_heatmap(time_slot, save_dir, is_show)
        elif option == 'network':
            self.ecs.plt_network_topology_graph(save_dir, is_show)
        elif option == 'deploy':
            self.function_deploy_stats.plt_function_instance_num_heatmap(time_slot, save_dir, '', is_show)
        else:
            raise TypeError("option can only be 'tasks', 'network' or 'deploy'.")

    def save_env(self, option='sys_info', time_slot=None, save_dir='', save_npy=False):
        """

        Args:
            option (str): 'sys_info'
            time_slot (int):
            save_dir (str): 保存路径
            save_npy (bool): 是否另存为 .npy 格式
        """
        if option == 'sys_info':
            self.ecs.save_sys_info(time_slot, save_dir, save_npy)
        else:
            raise TypeError("option can only be 'sys_info'.")
