import logging
import os
import threading
import time
from datetime import datetime

import numpy as np

from module.config.config import AzurLaneConfig
from module.log_res.log_res import LogRes
from module.statistics.cl1_database import db
from module.statistics.ship_exp_stats import get_ship_exp_stats

class OSSimulator:
    def __init__(self):
        self._init_logger()
        self._thread = None
        self._stop_event = threading.Event()

    def _init_logger(self):
        self.logger_path = f'./log/oss/{datetime.now().strftime("%Y-%m-%d")}.log'
        self.logger = logging.getLogger('OSSimulator')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            os.makedirs('./log/oss', exist_ok=True)
            fh = logging.FileHandler(self.logger_path, encoding='utf-8')
            fh.setFormatter(logging.Formatter(
                fmt='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(fh)
    
    AP = 0
    COIN = 1
    STATUS = 2
    USED_TIME = 3
    HAS_CRASHED = 4
    HAS_EARNED_COIN = 5
    MEOW_COUNT = 6
    CL1_COUNT = 7
    PASSED_DAYS = 8
    TIMESTAMP = 9
    
    STATUS_CL1 = 0
    STATUS_MEOW = 1
    STATUS_CRASHED = 2
    STATUS_DONE = 3
    
    AKASHI = np.array([20, 40, 50, 100, 100, 200] + [0] * 22)
    
    AP_RECOVER = 1 / 600
    AP_COSTS = {
        1: 5,
        2: 10,
        3: 15,
        4: 20,
        5: 30,
        6: 40
    }
    
    def _get_azurstat_data(self):
        # 预计之后使用azurstat统计数据，目前先这样吧（
        
        # 目前包括吊机
        hazard_level = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.MeowHazardLevel', 'level5')
        if hazard_level == 'level3':
            self.meow_hazard_level = 3
        else:
            self.meow_hazard_level = 5

        cl1_coin = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.Cl1Coin', 170)
        meow3_coin = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.Meow3Coin', 750)
        meow5_coin = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.Meow5Coin', 1700)
        self.coin_expectation = {
            1: cl1_coin,
            3: meow3_coin,
            5: meow5_coin
        }
        self.logger.info(f'每轮对应侵蚀等级期望获得黄币: {self.coin_expectation}')
        
        self.akashi_probability = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.AkashiProbability', 0.05)
        self.logger.info(f'遇见明石概率: {self.akashi_probability}')

        self.daily_reward = 6520
        self.logger.info(f'每日任务获得黄币: {self.daily_reward}')
        
        self.stronghold_reward = 40000
        self.logger.info(f'每周要塞期望获得黄币: {self.stronghold_reward}')
    
    def get_paras(self):
        self.config.load()
        
        self.samples = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.Samples')
        self.logger.info(f'样本数: {self.samples}')
        
        self.total_time = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.TotalTime')
        if not self.total_time:
            self.total_time = self._get_remaining_seconds()
        self.logger.info(f'总模拟时间 (s): {self.total_time}')
        
        self.time_use_ratio = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.TimeUseRatio')
        self.logger.info(f'时间利用率: {self.time_use_ratio}')
        
        self._get_azurstat_data() # 在 get_paras 中调用，确保初始化的 self.meow_hazard_level 能被后续使用
        self.logger.info(f'短猫侵蚀等级: {self.meow_hazard_level}')
        
        log_res = LogRes(self.config)
        ap = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.InitialAp')
        if not ap:
            ap = log_res.group('ActionPoint')
            ap = ap['Total'] if ap and 'Total' in ap else 0
        coin = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.InitialCoin')
        if not coin:
            coin = log_res.group('YellowCoin')
            coin = coin['Value'] if coin and 'Value' in coin else 0

        self.initial_state = np.array([
            np.ones(self.samples) * ap,    # ap
            np.ones(self.samples) * coin, # coin
            np.zeros(self.samples), # status (cl1: 0, meow: 1, crashed: 2, done: 3)
            np.zeros(self.samples), # used_time
            np.zeros(self.samples), # has_crashed
            np.zeros(self.samples), # has_earned_coin
            np.zeros(self.samples), # meow_count
            np.zeros(self.samples), # cl1_count
            np.zeros(self.samples), # passed_days
            np.ones(self.samples) * time.time(), # timestamp
        ])
        self.logger.info(f'初始黄币: {coin}')
        self.logger.info(f'初始行动力: {ap}')
        
        self.coin_preserve = self.config.cross_get('OpsiScheduling.OpsiScheduling.OperationCoinsPreserve')
        self.logger.info(f'保留黄币: {self.coin_preserve}')
        self.ap_preserve = self.config.cross_get('OpsiScheduling.OpsiScheduling.ActionPointPreserve')
        self.logger.info(f'保留行动力: {self.ap_preserve}')
        self.coin_threshold = self.config.cross_get('OpsiScheduling.OpsiScheduling.OperationCoinsReturnThreshold')
        self.logger.info(f'短猫直到获得多少黄币: {self.coin_threshold}')
        
        self.instance_name = getattr(self.config, 'config_name', 'default')
        self.logger.info(f'实例名: {self.instance_name}')
        
        if self.meow_hazard_level == 3:
            self.meow_time = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.Meow3Time', 0)
            if not self.meow_time:
                # 尝试从数据库获取短猫统计，如果不区分等级则统一使用平均值
                self.meow_time = db.get_meow_stats(self.instance_name).get('avg_round_time', 100)
        else: # hazard level 5
            self.meow_time = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.Meow5Time', 0)
            if not self.meow_time:
                self.meow_time = db.get_meow_stats(self.instance_name).get('avg_round_time', 200)
        
        self.logger.info(f'每轮短猫时间: {self.meow_time}')

        self.cl1_time = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.Cl1Time', 0)
        if not self.cl1_time:
            self.cl1_time = get_ship_exp_stats(self.instance_name).get_average_round_time()
        self.logger.info(f'每轮侵蚀1时间: {self.cl1_time}')
        
        # 修正后的单轮时间：包含了因“时间利用率”不足而产生的空闲时间，用于正确计算AP的自然恢复
        self.modified_meow_time = self.meow_time / self.time_use_ratio
        self.modified_cl1_time = self.cl1_time / self.time_use_ratio

        self.days_until_next_monday = self._get_days_until_next_monday()
        self.logger.info(f'距离下周一还有多少天: {self.days_until_next_monday}')

        # 调试模式开关：设置为 True 则取消随机性，按照期望值计算演化
        self.deterministic = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.Deterministic', False)
        
        if self.deterministic:
            self.samples = 1
            self.logger.info('调试模式：使用确定性计算。采样数已强制设置为 1 以提高计算速度。')
            self.logger.info('（不使用随机概率，按期望演化）')
        
        self.buy_ap = self.config.cross_get('OpsiSimulator.OpsiSimulatorParameters.BuyAp', True)
        self.logger.info(f'每周是否购买行动力: {self.buy_ap}')
    
    @property
    def is_running(self):
        return bool(self._thread and self._thread.is_alive())
    
    def _run(self):
        try:
            if not self.config:
                raise ValueError('缺少配置')
            
            self.get_paras()

            if self.meow_hazard_level not in self.coin_expectation:
                raise ValueError(f'不支持的短猫侵蚀等级: {self.meow_hazard_level}')

            self.logger.info("开始模拟...")
            start_time = time.time()
            result = self.simulate()
            self.logger.info(f"模拟完成，用时: {time.time() - start_time:.2f}秒")
            self._handle_result(result)
        except Exception as e:
            self.logger.exception(f"运行中出现错误: {e}")
    
    def set_config(self, config: AzurLaneConfig):
        self.config = config

    def start(self):
        if self.is_running:
            self.logger.warning("模拟正在进行，请耐心等待")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
    
    def interrupt(self):
        if self.is_running:
            self.logger.info("等待模拟中断")
            self._stop_event.set()
        else:
            self.logger.info("无正在进行的模拟")

    def _get_remaining_seconds(self):
        now = datetime.now()
        if now.month == 12:
            next_month = datetime(now.year + 1, 1, 1)
        else:
            next_month = datetime(now.year, now.month + 1, 1)
        return (next_month - now).total_seconds()

    def _get_days_until_next_monday(self):
        now = datetime.now()
        current_weekday = now.weekday()
        days_ahead = 7 - current_weekday
        return days_ahead
    
    def _handle_akashi(self, state, base_mask):
        n = np.sum(base_mask)
        if n == 0:
            return

        if self.deterministic:
            # 确定性模式：直接加上期望值
            # 期望 = 遇见的概率 * 购买的项数 * 奖池平均奖励
            reward_avg = np.mean(self.AKASHI)
            total_reward_expect = self.akashi_probability * 6 * reward_avg
            
            state[self.AP][base_mask] += total_reward_expect
            state[self.COIN][base_mask] -= total_reward_expect * 40
            return

        # Create a random mask for Akashi encounters, which is a subset of the base_mask
        rand_array = np.random.rand(state.shape[1])
        akashi_mask = (rand_array < self.akashi_probability) & base_mask
        n_akashi = np.sum(akashi_mask)
        
        if n_akashi == 0:
            return
        
        # 不放回采样28个里选6个 （我们numpy真是太厉害了）
        rand_mat = np.random.rand(28, n_akashi)
        indices = np.argpartition(rand_mat, 6, axis=0)[:6, :]
        sampled_values = self.AKASHI[indices]
        result = sampled_values.sum(axis=0)
        
        # 这里默认不会有人买不起行动力（不会吧？
        state[self.AP][akashi_mask] += result
        state[self.COIN][akashi_mask] -= result * 40
        
    def _cl1_simulate(self, state, mask):
        state[self.CL1_COUNT][mask] += 1
        
        state[self.AP][mask] -= self.AP_COSTS[1]
        state[self.COIN][mask] += self.coin_expectation[1]
        
        state[self.AP][mask] += self.AP_RECOVER * self.modified_cl1_time
        state[self.USED_TIME][mask] += self.modified_cl1_time
        state[self.TIMESTAMP][mask] += self.modified_cl1_time
        
        self._handle_akashi(state, mask)
    
    def _meow_simulate(self, state, mask):
        state[self.MEOW_COUNT][mask] += 1
        
        state[self.AP][mask] -= self.AP_COSTS[self.meow_hazard_level]
        state[self.COIN][mask] += self.coin_expectation[self.meow_hazard_level]
        state[self.HAS_EARNED_COIN][mask] += self.coin_expectation[self.meow_hazard_level]
        
        state[self.AP][mask] += self.AP_RECOVER * self.modified_meow_time
        state[self.USED_TIME][mask] += self.modified_meow_time
        state[self.TIMESTAMP][mask] += self.modified_meow_time
        
        self._handle_akashi(state, mask)
    
    def _crashed_simulate(self, state, mask):
        state[self.HAS_CRASHED][mask] = 1
        skip_time = 43200   # 12 * 60 * 60
        
        state[self.USED_TIME][mask] += skip_time
        state[self.TIMESTAMP][mask] += skip_time
        state[self.AP][mask] += 72
    
    def simulate(self):
        if not hasattr(self, 'initial_state'):
            self.get_paras()
            
        now_state = np.copy(self.initial_state)
        # 记录历史数据，Shape: (Steps, 2, Samples)
        history = []
        
        while np.any(now_state[self.STATUS] != self.STATUS_DONE):
            if self._stop_event.is_set():
                self.logger.info("模拟中断")
                break

            # 记录当前平均状态
            history.append([np.copy(now_state[self.AP]), np.copy(now_state[self.COIN]), np.copy(now_state[self.TIMESTAMP])])

            # 1. 计算状态转移
            is_cl1 = now_state[self.STATUS] == self.STATUS_CL1
            is_meow = now_state[self.STATUS] == self.STATUS_MEOW
            is_crashed = now_state[self.STATUS] == self.STATUS_CRASHED
            
            # 从侵蚀1切换到短猫
            # 触发条件：当前处于侵蚀1，且黄币跌破保留值
            to_meow_mask = is_cl1 & (now_state[self.COIN] < self.coin_preserve)
            
            # 从短猫切换到侵蚀1
            # 触发条件：当前处于短猫，且黄币由于短猫补充后，已经超过了 (保留值 + 单次目标值)
            to_cl1_mask = is_meow & (now_state[self.COIN] >= (self.coin_preserve + self.coin_threshold))
            
            # 从坠机切换到侵蚀1
            # 触发条件：当前处于坠机等待，行动力自然恢复到了允许执行侵蚀1的水平（至少5点）
            to_cl1_mask |= is_crashed & (now_state[self.AP] >= self.AP_COSTS[1])
            
            # 从侵蚀1或者短猫切换到坠机
            # 触发条件：
            # 1. 连侵蚀1都跑不起了（AP < 5）
            # 2. 黄币低于保留值需要补猫，但 AP 也低于保留值（没钱也没豆，且金币不能为负，无法购买 AP）
            to_crashed_mask = (is_cl1 | is_meow) & (now_state[self.AP] < self.AP_COSTS[1])
            to_crashed_mask |= is_cl1 & (now_state[self.COIN] < self.coin_preserve) & (now_state[self.AP] < self.ap_preserve)
            
            # 从侵蚀1切换到短猫的额外限制：只有在 AP 高于保留值时才允许切换去短猫
            to_meow_mask = is_cl1 & (now_state[self.COIN] < self.coin_preserve) & (now_state[self.AP] >= self.ap_preserve)
            
            # 如果正在短猫但 AP 掉到了保留值以下，强制切回侵蚀1（即便金币还没攒够）
            to_cl1_mask |= is_meow & (now_state[self.AP] < self.ap_preserve)
            
            # 2. 应用状态转移
            now_state[self.STATUS][to_meow_mask] = self.STATUS_MEOW
            now_state[self.HAS_EARNED_COIN][to_meow_mask] = 0
            now_state[self.STATUS][to_cl1_mask] = self.STATUS_CL1
            now_state[self.STATUS][to_crashed_mask] = self.STATUS_CRASHED
            
            # 3. 执行模拟步进
            for status_val, sim_func in zip([self.STATUS_CL1, self.STATUS_MEOW, self.STATUS_CRASHED], [self._cl1_simulate, self._meow_simulate, self._crashed_simulate]):
                mask = now_state[self.STATUS] == status_val
                if np.any(mask):
                    sim_func(now_state, mask)

            # 4. 更新跨日
            sim_days = now_state[self.USED_TIME] // 86400
            cross_day_mask = sim_days > now_state[self.PASSED_DAYS]
            if np.any(cross_day_mask):
                now_state[self.PASSED_DAYS][cross_day_mask] += 1
                now_state[self.COIN][cross_day_mask] += self.daily_reward

                cross_week_mask = (sim_days - self.days_until_next_monday) % 7 == 0
                if np.any(cross_week_mask & cross_day_mask):
                    if self.buy_ap:
                        now_state[self.AP][cross_day_mask & cross_week_mask] += 800
                    else:
                        now_state[self.AP][cross_day_mask & cross_week_mask] -= 200
                    now_state[self.COIN][cross_day_mask & cross_week_mask] += self.stronghold_reward
            
            # 5. 标记完成状态
            now_state[self.STATUS][now_state[self.USED_TIME] >= self.total_time] = self.STATUS_DONE
            
        return now_state, np.array(history)
    
    def _handle_result(self, result):
        import numpy as np
        result, history = result
        self.result_cl1_count = np.average(result[self.CL1_COUNT])
        self.logger.info(f'[模拟结果] 侵蚀1次数: {self.result_cl1_count}')
        self.result_meow_count = np.average(result[self.MEOW_COUNT])
        self.logger.info(f'[模拟结果] 短猫次数: {self.result_meow_count}')
        self.result_crashed_probability = np.average(result[self.HAS_CRASHED])
        self.logger.info(f'[模拟结果] 坠机概率: {self.result_crashed_probability}')
        self.result_cl1_total_time = self.result_cl1_count * self.cl1_time
        self.logger.info(f'[模拟结果] 侵蚀一总时长 (h): {self.result_cl1_total_time / 3600}')
        self.result_meow_total_time = self.result_meow_count * self.meow_time
        self.logger.info(f'[模拟结果] 短猫总时长 (h): {self.result_meow_total_time / 3600}')

        if self.deterministic:
            self.result_ap = np.average(result[self.AP])
            self.logger.info(f'[模拟结果] 最终行动力: {self.result_ap}')
            self.result_coin = np.average(result[self.COIN])
            self.logger.info(f'[模拟结果] 最终黄币: {self.result_coin}')
        else:
            # 获取样本总数，防止请求的 5 个超出范围
            n_samples = result.shape[1]
            top_k = min(5, n_samples)
            
            # 找到 AP 最低的 5 个索引
            # np.argsort 默认升序，即前 5 个是最小的
            bottom_indices = np.argsort(result[self.AP])[:top_k]
            # 找到 AP 最高的 5 个索引
            top_indices = np.argsort(result[self.AP])[-top_k:][::-1]
            
            self.logger.info(f'[模拟结果] 最差情况 (AP最低的前 {top_k} 个样本):')
            for i, idx in enumerate(bottom_indices):
                self.logger.info(f'  No.{i+1}: AP {result[self.AP][idx]:.1f}, Coin {result[self.COIN][idx]:.0f}')
            
            self.logger.info(f'[模拟结果] 最好情况 (AP最高的后 {top_k} 个样本):')
            for i, idx in enumerate(top_indices):
                self.logger.info(f'  No.{i+1}: AP {result[self.AP][idx]:.1f}, Coin {result[self.COIN][idx]:.0f}')

        # 绘制折线图
        try:
            import numpy as np
            import os
            import sys
            import subprocess
            from datetime import datetime
            
            if not self.deterministic:
                self.logger.info("非确定性模式，跳过绘制折线图（平均值在概率模拟下无意义）")
                return

            # 计算所有样本的平均值
            avg_history = np.mean(history, axis=2) # Shape: (Steps, 3)
            # steps = np.arange(len(avg_history)) # 不再需要步数作为 X 轴
            timestamps = avg_history[:, 2] # 获取时间戳作为 X 轴
            
            os.makedirs('./log/oss', exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            csv_path = f'./log/oss/{timestamp_str}.csv'
            
            # 保存到 CSV (每行: Timestamp, AP_Avg, Coin_Avg)
            header = "Timestamp,AP,Coin"
            csv_data = np.column_stack((timestamps, avg_history[:, 0], avg_history[:, 1]))
            np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments='')
            
            # 调用子脚本绘图 (异步)
            python_exe = sys.executable
            subprocess.Popen([python_exe, 'module/os/draw_os_plot.py', csv_path, str(self.coin_preserve), str(self.coin_threshold)])
            
            self.logger.info(f"数据已保存至 CSV，绘图子进程已启动: {csv_path}")
        except Exception as e:
            self.logger.error(f"保存或启动绘图失败: {e}")