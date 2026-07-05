"""AMCL lifecycle 控制（subprocess 调用 ros2 CLI）。"""

from __future__ import annotations

import subprocess
from typing import Callable, Optional


LogFn = Callable[[str], None]
WarnFn = Callable[[str], None]


class AmclLifecycle:
    def __init__(
        self,
        amcl_node_name: str,
        lifecycle_manager_name: str,
        cmd_timeout: float,
        log_info: LogFn,
        log_warn: WarnFn,
    ) -> None:
        self._amcl = amcl_node_name
        self._manager = lifecycle_manager_name
        self._timeout = cmd_timeout
        self._log_info = log_info
        self._log_warn = log_warn

    def _run(self, args: list, label: str) -> bool:
        try:
            proc = subprocess.run(
                ['ros2', *args],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or '').strip()
                self._log_warn(f'{label} failed: {err}')
                return False
            self._log_info(f'{label} OK')
            return True
        except subprocess.TimeoutExpired:
            self._log_warn(f'{label} timeout')
            return False
        except OSError as exc:
            self._log_warn(f'{label} error: {exc}')
            return False

    def set_tf_broadcast(self, enabled: bool) -> bool:
        val = 'true' if enabled else 'false'
        return self._run([
            'param', 'set', f'/{self._amcl}', 'tf_broadcast', val,
        ], f'amcl tf_broadcast={val}')

    def pause_via_manager(self) -> bool:
        return self._run([
            'service', 'call',
            f'/{self._manager}/manage_nodes',
            'nav2_msgs/srv/ManageLifecycleNodes',
            '{command: 1}',
        ], 'lifecycle_manager PAUSE')

    def resume_via_manager(self) -> bool:
        return self._run([
            'service', 'call',
            f'/{self._manager}/manage_nodes',
            'nav2_msgs/srv/ManageLifecycleNodes',
            '{command: 2}',
        ], 'lifecycle_manager RESUME')

    def deactivate(self) -> bool:
        return self._run([
            'lifecycle', 'set', f'/{self._amcl}', 'deactivate',
        ], 'amcl deactivate')

    def activate(self) -> bool:
        return self._run([
            'lifecycle', 'set', f'/{self._amcl}', 'activate',
        ], 'amcl activate')

    def pause(self) -> bool:
        if self.set_tf_broadcast(False):
            self._log_info('AMCL tf_broadcast=false')
        if self.pause_via_manager():
            self._log_info('AMCL 已通过 lifecycle_manager PAUSE')
            return True
        if self.deactivate():
            self._log_info('AMCL 已 lifecycle deactivate')
            return True
        self._log_warn('AMCL lifecycle pause 失败，已依赖 tf_broadcast=false')
        return self.set_tf_broadcast(False)

    def resume(self) -> bool:
        self.set_tf_broadcast(True)
        if self.resume_via_manager():
            self._log_info('AMCL 已通过 lifecycle_manager RESUME')
            return True
        if self.activate():
            self._log_info('AMCL 已 lifecycle activate')
            return True
        self._log_warn('AMCL 未能自动恢复，请手动 activate')
        return False
