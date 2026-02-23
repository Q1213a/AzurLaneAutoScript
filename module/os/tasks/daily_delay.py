"""
OpsiDailyDelay - 大世界任务延后模块

在每日0点服务器重启前的可配置时间段内，将所有当前进行中的大世界任务自动延后至次日0点重启完成后再执行。

功能说明:
    1. 定时触发 - 在每日0点前X分钟自动触发（X可配置，范围1-60分钟，默认5分钟）
    2. 任务过滤 - 自动过滤非大世界任务、跨月任务和紧急任务
    3. 任务延后 - 将符合条件的大世界任务延后到0点后执行
    4. 状态暂存 - 保存任务状态，支持断点续传
    5. 日志记录 - 完整的日志记录功能，便于问题排查

任务层级:
    - OpsiDailyDelay 是和 OpsiCrossMonth 相同层级的调度器
    - 它负责在每日0点前延后大世界任务，避免任务在服务器重启时中断

配置项:
    - Scheduler.Enable: 任务启用开关（启用此任务即启用大世界任务延后功能）
    - OpsiDailyDelay.Enable: 功能总开关（默认禁用）
    - OpsiDailyDelay.TriggerMinutesBeforeReset: 提前触发时间（分钟，默认5，范围1-60）

此模块包含:
    - OpsiDailyDelay: 大世界任务延后任务主类
"""
import json
import os
from datetime import datetime, timedelta

from module.config.utils import get_server_next_update
from module.exception import ScriptError
from module.logger import logger
from module.os.map import OSMap


class OpsiDailyDelay(OSMap):
    """
    大世界任务延后任务主类
    
    功能:
    - 在每日0点前X分钟自动触发
    - 延后所有进行中的大世界任务到0点后
    - 自动过滤非大世界任务、跨月任务和紧急任务
    - 保存任务状态，支持断点续传
    """
    
    # 紧急任务列表（不会被延后）
    EMERGENCY_TASKS = [
        'OpsiAshBeacon',  # 灰烬信标（紧急任务）
    ]
    
    # 任务状态文件路径
    STATUS_FILE = './log/opsi_daily_delay_status.json'
    
    # 配置路径常量
    CONFIG_PATH_ENABLE = 'OpsiDailyDelay.OpsiDailyDelay.Enable'
    CONFIG_PATH_TRIGGER_MINUTES = 'OpsiDailyDelay.OpsiDailyDelay.TriggerMinutesBeforeReset'
    
    # ==================== 时间计算相关方法 ====================
    
    def _calculate_trigger_time(self, minutes_before_reset):
        """
        计算触发时间（0点前X分钟）
        
        Args:
            minutes_before_reset: 提前分钟数
            
        Returns:
            datetime: 触发时间（本地时间）
        """
        # 获取下次0点时间（服务器时间）
        next_reset = get_server_next_update("00:00")
        
        # 计算触发时间（0点前X分钟）
        trigger_time = next_reset - timedelta(minutes=minutes_before_reset)
        
        logger.info(f'计算触发时间: 0点前{minutes_before_reset}分钟，触发时间={trigger_time}')
        
        return trigger_time
    
    def _calculate_recovery_time(self):
        """
        计算恢复时间（0点后）
        
        Returns:
            datetime: 恢复时间（本地时间）
        """
        # 获取下次0点时间（服务器时间）
        next_reset = get_server_next_update("00:00")
        
        # 恢复时间为0点后5分钟
        recovery_time = next_reset + timedelta(minutes=5)
        
        logger.info(f'计算恢复时间: 0点后5分钟，恢复时间={recovery_time}')
        
        return recovery_time
    
    # ==================== 任务过滤相关方法 ====================
    
    def _should_delay_task(self, task_name):
        """
        判断任务是否应该被延后
        
        排除规则:
        1. 非大世界任务（不以Opsi开头）
        2. 跨月任务（OpsiCrossMonth）
        3. 紧急任务（在EMERGENCY_TASKS列表中）
        4. 自身任务（OpsiDailyDelay）
        
        Args:
            task_name: 任务名称
            
        Returns:
            bool: True表示应该延后，False表示不应该延后
        """
        # 排除非大世界任务
        if not task_name.startswith('Opsi'):
            logger.debug(f'任务过滤: {task_name} - 非大世界任务，跳过')
            return False
        
        # 排除跨月任务
        if task_name == 'OpsiCrossMonth':
            logger.debug(f'任务过滤: {task_name} - 跨月任务，跳过')
            return False
        
        # 排除紧急任务
        if task_name in self.EMERGENCY_TASKS:
            logger.debug(f'任务过滤: {task_name} - 紧急任务，跳过')
            return False
        
        # 排除自身
        if task_name == 'OpsiDailyDelay':
            logger.debug(f'任务过滤: {task_name} - 自身任务，跳过')
            return False
        
        logger.debug(f'任务过滤: {task_name} - 符合条件，将延后')
        return True
    
    # ==================== 任务状态管理相关方法 ====================
    
    def _save_task_status(self, task_name, original_next_run, delayed_next_run):
        """
        保存任务状态
        
        Args:
            task_name: 任务名称
            original_next_run: 原始NextRun时间
            delayed_next_run: 延后后的NextRun时间
        """
        # 读取现有状态
        status = {}
        if os.path.exists(self.STATUS_FILE):
            try:
                with open(self.STATUS_FILE, 'r', encoding='utf-8') as f:
                    status = json.load(f)
            except Exception as e:
                logger.warning(f'读取任务状态文件失败: {e}，将创建新文件')
        
        # 保存任务状态
        status[task_name] = {
            'original_next_run': original_next_run.strftime('%Y-%m-%d %H:%M:%S'),
            'delayed_next_run': delayed_next_run.strftime('%Y-%m-%d %H:%M:%S'),
            'delayed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 写入文件
        try:
            with open(self.STATUS_FILE, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
            logger.info(f'保存任务状态: {task_name}')
        except Exception as e:
            logger.error(f'保存任务状态失败: {e}')
    
    def _load_task_status(self, task_name):
        """
        加载任务状态
        
        Args:
            task_name: 任务名称
            
        Returns:
            dict: 任务状态，如果不存在则返回None
        """
        # 检查文件是否存在
        if not os.path.exists(self.STATUS_FILE):
            return None
        
        # 读取状态
        try:
            with open(self.STATUS_FILE, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return status.get(task_name)
        except Exception as e:
            logger.error(f'读取任务状态失败: {e}')
            return None
    
    def _clear_task_status(self, task_name=None):
        """
        清除任务状态
        
        Args:
            task_name: 任务名称，如果为None则清除所有状态
        """
        # 检查文件是否存在
        if not os.path.exists(self.STATUS_FILE):
            return
        
        if task_name is None:
            # 清除所有状态
            try:
                os.remove(self.STATUS_FILE)
                logger.info('清除所有任务状态')
            except Exception as e:
                logger.error(f'清除任务状态失败: {e}')
        else:
            # 清除指定任务状态
            try:
                with open(self.STATUS_FILE, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                
                if task_name in status:
                    del status[task_name]
                    logger.info(f'清除任务状态: {task_name}')
                
                with open(self.STATUS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(status, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f'清除任务状态失败: {e}')
    
    # ==================== 任务延后相关方法 ====================
    
    def _delay_task(self, task_name, delay_to):
        """
        延后任务到指定时间
        
        Args:
            task_name: 任务名称
            delay_to: 延后到的时间（datetime对象）
            
        Returns:
            bool: 是否成功延后
        """
        try:
            # 获取任务原始NextRun时间
            original_next_run = self.config.cross_get(keys=f'{task_name}.Scheduler.NextRun')
            
            if not isinstance(original_next_run, datetime):
                logger.warning(f'任务 {task_name} 的 NextRun 不是 datetime 对象，跳过')
                return False
            
            # 保存任务状态
            self._save_task_status(task_name, original_next_run, delay_to)
            
            # 设置延后时间
            self.config.cross_set(keys=f'{task_name}.Scheduler.NextRun', value=delay_to)
            
            logger.info(f'延后任务: {task_name} 从 {original_next_run} 到 {delay_to}')
            
            return True
        except Exception as e:
            logger.error(f'延后任务失败: {task_name}, 错误: {e}')
            return False
    
    def _delay_all_pending_tasks(self, delay_to):
        """
        延后所有待执行的大世界任务
        
        Args:
            delay_to: 延后到的时间（datetime对象）
            
        Returns:
            int: 成功延后的任务数量
        """
        # 获取所有待执行任务
        pending_tasks = self.config.pending_task
        waiting_tasks = self.config.waiting_task
        
        # 合并所有任务
        all_tasks = pending_tasks + waiting_tasks
        
        if not all_tasks:
            logger.info('没有待执行的任务')
            return 0
        
        logger.info(f'待执行任务数量: {len(all_tasks)}')
        
        # 延后符合条件的任务
        delayed_count = 0
        for task in all_tasks:
            task_name = task.command
            
            # 检查是否应该延后
            if self._should_delay_task(task_name):
                if self._delay_task(task_name, delay_to):
                    delayed_count += 1
        
        logger.info(f'成功延后任务数量: {delayed_count}')
        
        return delayed_count
    
    # ==================== 任务恢复相关方法 ====================
    
    def _restore_task(self, task_name):
        """
        恢复任务到原始时间
        
        Args:
            task_name: 任务名称
            
        Returns:
            bool: 是否成功恢复
        """
        try:
            # 加载任务状态
            status = self._load_task_status(task_name)
            
            if not status:
                logger.warning(f'任务 {task_name} 没有保存的状态，跳过')
                return False
            
            # 解析原始时间
            original_next_run = datetime.strptime(status['original_next_run'], '%Y-%m-%d %H:%M:%S')
            
            # 恢复原始时间
            self.config.cross_set(keys=f'{task_name}.Scheduler.NextRun', value=original_next_run)
            
            logger.info(f'恢复任务: {task_name} 到 {original_next_run}')
            
            # 清除任务状态
            self._clear_task_status(task_name)
            
            return True
        except Exception as e:
            logger.error(f'恢复任务失败: {task_name}, 错误: {e}')
            return False
    
    def _restore_all_delayed_tasks(self):
        """
        恢复所有延后的任务
        
        Returns:
            int: 成功恢复的任务数量
        """
        # 检查状态文件是否存在
        if not os.path.exists(self.STATUS_FILE):
            logger.info('没有延后的任务需要恢复')
            return 0
        
        # 读取所有任务状态
        try:
            with open(self.STATUS_FILE, 'r', encoding='utf-8') as f:
                status = json.load(f)
        except Exception as e:
            logger.error(f'读取任务状态失败: {e}')
            return 0
        
        if not status:
            logger.info('没有延后的任务需要恢复')
            return 0
        
        logger.info(f'延后任务数量: {len(status)}')
        
        # 恢复所有任务
        restored_count = 0
        for task_name in list(status.keys()):
            if self._restore_task(task_name):
                restored_count += 1
        
        logger.info(f'成功恢复任务数量: {restored_count}')
        
        return restored_count
    
    # ==================== 配置验证相关方法 ====================
    
    def _validate_trigger_minutes(self, minutes):
        """
        验证提前触发时间是否有效
        
        Args:
            minutes: 提前分钟数
            
        Returns:
            bool: 是否有效
        """
        if not isinstance(minutes, int):
            logger.error(f'提前触发时间必须是整数，当前值: {minutes}')
            return False
        
        if minutes < 1 or minutes > 60:
            logger.error(f'提前触发时间必须在1-60分钟之间，当前值: {minutes}')
            return False
        
        return True
    
    # ==================== 主任务方法 ====================
    
    def opsi_daily_delay_end(self):
        """
        任务结束处理
        
        延迟到下次触发时间（0点前X分钟）
        """
        # 获取提前触发时间
        trigger_minutes = self.config.cross_get(keys=self.CONFIG_PATH_TRIGGER_MINUTES)
        
        # 验证配置
        if not self._validate_trigger_minutes(trigger_minutes):
            logger.error('提前触发时间配置无效，使用默认值5分钟')
            trigger_minutes = 5
        
        # 计算下次触发时间
        next_trigger_time = self._calculate_trigger_time(trigger_minutes)
        
        # 延迟到下次触发时间
        self.config.task_delay(target=next_trigger_time)
        
        # 停止任务
        self.config.task_stop()
    
    def opsi_daily_delay(self):
        """
        大世界任务延后主任务
        
        执行流程:
        1. 检查功能是否启用
        2. 计算触发时间
        3. 等待到触发时间
        4. 延后所有待执行的大世界任务
        5. 等待到0点
        6. 恢复所有延后的任务
        7. 任务结束
        """
        logger.hr('大世界任务延后', level=1)
        
        # 检查功能是否启用
        enable = self.config.cross_get(keys=self.CONFIG_PATH_ENABLE)
        if not enable:
            logger.info('大世界任务延后功能未启用，跳过')
            self.opsi_daily_delay_end()
            return
        
        logger.info('大世界任务延后功能已启用')
        
        # 获取提前触发时间
        trigger_minutes = self.config.cross_get(keys=self.CONFIG_PATH_TRIGGER_MINUTES)
        
        # 验证配置
        if not self._validate_trigger_minutes(trigger_minutes):
            logger.error('提前触发时间配置无效，使用默认值5分钟')
            trigger_minutes = 5
        
        logger.info(f'提前触发时间: {trigger_minutes}分钟')
        
        # 计算触发时间
        trigger_time = self._calculate_trigger_time(trigger_minutes)
        now = datetime.now()
        
        # 检查触发时间
        if trigger_time < now:
            logger.error(f'触发时间 {trigger_time} 早于当前时间 {now}，跳过')
            self.opsi_daily_delay_end()
            return
        
        if trigger_time - now > timedelta(hours=24):
            logger.error(f'触发时间 {trigger_time} 距离当前时间超过24小时，跳过')
            self.opsi_daily_delay_end()
            return
        
        # 等待到触发时间
        logger.hr('等待到触发时间', level=1)
        logger.info(f'等待到 {trigger_time}')
        while True:
            now = datetime.now()
            remain = (trigger_time - now).total_seconds()
            
            if remain <= 0:
                break
            
            logger.info(f'剩余时间: {remain:.0f}秒')
            self.device.sleep(min(remain, 60))
        
        logger.hr('触发时间到达', level=2)
        
        # 计算恢复时间（0点后5分钟）
        recovery_time = self._calculate_recovery_time()
        
        # 延后所有待执行的大世界任务
        logger.hr('延后大世界任务', level=1)
        delayed_count = self._delay_all_pending_tasks(recovery_time)
        
        if delayed_count > 0:
            logger.info(f'成功延后 {delayed_count} 个大世界任务')
        else:
            logger.info('没有需要延后的大世界任务')
        
        # 等待到0点
        logger.hr('等待到0点', level=1)
        next_reset = get_server_next_update("00:00")
        logger.info(f'等待到 {next_reset}')
        while True:
            now = datetime.now()
            remain = (next_reset - now).total_seconds()
            
            if remain <= 0:
                break
            
            logger.info(f'剩余时间: {remain:.0f}秒')
            self.device.sleep(min(remain, 60))
        
        logger.hr('0点到达', level=2)
        
        # 等待5分钟，确保服务器重启完成
        logger.hr('等待服务器重启完成', level=1)
        logger.info('等待5分钟，确保服务器重启完成')
        self.device.sleep(300)
        
        # 恢复所有延后的任务
        logger.hr('恢复大世界任务', level=1)
        restored_count = self._restore_all_delayed_tasks()
        
        if restored_count > 0:
            logger.info(f'成功恢复 {restored_count} 个大世界任务')
        else:
            logger.info('没有需要恢复的大世界任务')
        
        logger.hr('大世界任务延后完成', level=1)
        
        # 任务结束
        self.opsi_daily_delay_end()
