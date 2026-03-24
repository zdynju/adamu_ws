"""
contact_phase.py — 双臂接触箱子前上边棱

核心设计原则
------------
探触、夹紧、保持三个阶段合并在一次 servo_both_cartesian 调用里。

原因：如果分成多次调用，每次调用结束时 servo_cartesian 的 finally
块会发零速并销毁定时器，Servo 节点超时后关节自由，夹紧力泄掉。

解决方法：
  - velocity_fn 内部维护每侧的阶段状态，自行切换
  - termination_fn 只处理异常（过载、总超时），不参与正常流转
  - 力的连续性由 Servo 持续运行保证，中间没有任何停顿

调用前提
--------
  - 双臂 Servo 已激活（activate_both_servo 已调用）
  - 两臂已通过 MoveIt move_group 到达预接触位置
    （末端在边棱外侧约 20~30mm，高度与边棱对齐）
  - fts_processor 已完成 tare（接触前零偏标定）
"""

import time
import asyncio
from dataclasses import dataclass

from adamu_manipulation.servo_controller import AdamuServoController


# ─────────────────────────────────────────────────────────────────────────────
#  参数
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContactParams:
    """
    接触阶段所有参数。

    调参顺序：
      1. search_speed 先设最小值（0.005），确认接触检测稳定
      2. contact_force_threshold 从大往小调，找到不被噪声误触发的最小值
      3. target_grasp_force 从小往大加，确认夹住不滑
      4. max_grasp_force 设为 target 的 1.5~2 倍作为硬件保护
    """

    # 探触速度（m/s），向中心推进，建议 0.005~0.02
    search_speed: float = 0.1

    # 夹紧速度（m/s），比探触慢，给力传感器响应时间，建议 0.002~0.005
    buildup_speed: float = 0.1

    # 接触检测阈值（N）：Y 方向力超过此值认为已接触边棱
    # 太小：传感器噪声误触发；太大：已经压进去了
    contact_force_threshold: float = 3.0

    # 目标夹紧力（N）：翻转开始前需要建立的初始内力
    # 估算：F > m*g / (2*mu)，m=箱子质量，mu=摩擦系数
    target_grasp_force: float = 15.0

    # 过载保护上限（N）：内力超过此值立即停止，防止压坏箱子
    # 建议设为 target_grasp_force 的 1.5~2 倍
    max_grasp_force: float = 30.0

    # 整个接触阶段的总超时（s），包含探触和夹紧
    total_timeout_sec: float = 20.0

    # 控制频率（Hz）
    rate_hz: float = 100.0


# ─────────────────────────────────────────────────────────────────────────────
#  单侧阶段状态（velocity_fn 闭包内部共享）
# ─────────────────────────────────────────────────────────────────────────────

class _SidePhase:
    """
    单臂的阶段跟踪器。

    velocity_fn 是闭包，用这个对象在左右两侧的闭包之间
    共享状态——左侧的 velocity_fn 需要知道右侧是否已完成探触，
    才能决定是否开始推进夹紧。

    阶段流转：SEARCHING → BUILDING → HOLDING
    """
    SEARCHING = 'searching'
    BUILDING  = 'building'
    HOLDING   = 'holding'

    def __init__(self):
        self.phase = self.SEARCHING


# ─────────────────────────────────────────────────────────────────────────────
#  接触阶段
# ─────────────────────────────────────────────────────────────────────────────

class ContactPhase:

    def __init__(self, ctrl: AdamuServoController, params: ContactParams):
        self.ctrl   = ctrl
        self.params = params

    # ── 对外接口 ─────────────────────────────────────────────────────────────

    async def run(self) -> bool:
        """
        执行完整接触序列：探触 → 夹紧 → 保持。

        全程一次 servo_both_cartesian 调用，力不中断。

        返回 True：双臂夹紧力已建立，可以进入翻转阶段。
        返回 False：发生异常（超时、过载），上层应中止任务。
        """
        self.ctrl.get_logger().info('=== 接触阶段开始 ===')

        phase_left  = _SidePhase()
        phase_right = _SidePhase()
        start       = time.monotonic()

        ok = await self.ctrl.servo_both_cartesian(
            velocity_fn_left  = self._make_velocity_fn('left',  phase_left,  phase_right),
            velocity_fn_right = self._make_velocity_fn('right', phase_right, phase_left),
            termination_fn    = self._make_termination_fn(phase_left, phase_right, start),
            rate_hz=self.params.rate_hz,
        )

        if not ok:
            self.ctrl.get_logger().error('接触阶段异常退出')
            return False

        # 最终验证：termination_fn 超时退出时夹紧力可能不足
        f_int = self._internal_force()
        if f_int < self.params.target_grasp_force * 0.8:
            self.ctrl.get_logger().error(
                f'夹紧力不足: {f_int:.1f}N，'
                f'目标 {self.params.target_grasp_force:.1f}N'
            )
            return False

        self.ctrl.get_logger().info(f'=== 接触阶段完成，内力 {f_int:.1f}N ===')
        return True

    # ── velocity_fn 工厂 ─────────────────────────────────────────────────────

    def _make_velocity_fn(
        self,
        side: str,
        my_phase: _SidePhase,
        other_phase: _SidePhase,
    ):
        """
        生成单侧的 velocity_fn。

        阶段逻辑
        --------
        SEARCHING（探触）
          以 search_speed 向中心推进。
          检测本侧 Y 方向力，超过阈值则切换到 BUILDING。
          若本侧已切换 BUILDING 但对侧仍在 SEARCHING，
          则发零速等待——避免一侧过度推进导致不对称夹紧。

        BUILDING（夹紧）
          双臂都已接触，以 buildup_speed 继续推进。
          检测内力（两侧均值）：
            超过 max_grasp_force → 过载，停止推进，等 termination_fn 退出
            达到 target_grasp_force → 切换到 HOLDING

        HOLDING（保持）
          发零速，Servo 维持当前关节位置，夹紧力由末端位置保持。
          termination_fn 检测到双侧都 HOLDING 后正常退出。
        """
        v_sign = -1.0 if side == 'left' else 1.0
        p      = self.params
        logger = self.ctrl.get_logger()

        def velocity_fn(t: float) -> list:

            # ── SEARCHING ────────────────────────────────────────────────────
            if my_phase.phase == _SidePhase.SEARCHING:

                f_y = self._contact_force_y(side)
                if abs(f_y) >= p.contact_force_threshold:
                    my_phase.phase = _SidePhase.BUILDING
                    logger.info(f'[{side}] 接触确认 F_y={f_y:.2f}N → BUILDING')
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                print(v_sign * p.search_speed)
                return [0.0, v_sign * p.search_speed, 0.0, 0.0, 0.0, 0.0]

            # ── BUILDING ─────────────────────────────────────────────────────
            if my_phase.phase == _SidePhase.BUILDING:

                # 对侧还在探触，等待——不能一侧开始夹紧另一侧还没接触
                if other_phase.phase == _SidePhase.SEARCHING:
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                f_int = self._internal_force()

                if f_int > p.max_grasp_force:
                    # 过载，停止推进，termination_fn 会处理退出
                    logger.error(
                        f'[{side}] 过载 内力={f_int:.1f}N > {p.max_grasp_force}N'
                    )
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                if f_int >= p.target_grasp_force:
                    my_phase.phase = _SidePhase.HOLDING
                    logger.info(f'[{side}] 夹紧力达标 内力={f_int:.1f}N → HOLDING')
                    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                return [0.0, v_sign * p.buildup_speed, 0.0, 0.0, 0.0, 0.0]

            # ── HOLDING ──────────────────────────────────────────────────────
            # 零速，Servo 维持位置，夹紧力靠末端位置保持
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return velocity_fn

    # ── termination_fn 工厂 ──────────────────────────────────────────────────

    def _make_termination_fn(
        self,
        phase_left: _SidePhase,
        phase_right: _SidePhase,
        start: float,
    ):
        """
        生成 termination_fn。

        只处理三种退出情况：
          1. 双侧都 HOLDING → 正常结束
          2. 严重过载（velocity_fn 已停推进，这里触发退出）
          3. 总超时

        不包含夹紧力判断。夹紧力判断在 velocity_fn 里做，
        在 termination_fn 里判断夹紧力会导致 Servo 停止从而力泄掉。
        """
        p      = self.params
        logger = self.ctrl.get_logger()

        def termination_fn() -> bool:

            # 正常结束
            if (phase_left.phase  == _SidePhase.HOLDING and
                phase_right.phase == _SidePhase.HOLDING):
                return True

            # 严重过载（乘 1.2 留裕量，让 velocity_fn 先处理一帧）
            f_int = self._internal_force()
            if f_int > p.max_grasp_force * 1.2:
                logger.error(f'严重过载 {f_int:.1f}N，紧急退出')
                return True

            # 总超时
            if time.monotonic() - start > p.total_timeout_sec:
                logger.warn(
                    f'接触阶段总超时 {p.total_timeout_sec}s，'
                    f'内力={f_int:.1f}N，'
                    f'左={phase_left.phase} 右={phase_right.phase}'
                )
                return True

            return False

        return termination_fn

    # ── 力传感器辅助 ─────────────────────────────────────────────────────────

    def _contact_force_y(self, side: str) -> float:
        """
        读取指定侧 Y 方向（夹紧方向）的接触力（N）。
        fts_processor 在接触前已 tare，读到的是纯接触力。
        """
        return self.ctrl.fts_processor.get_standardized_force(side)[1]

    def _internal_force(self) -> float:
        """
        计算当前内力（夹紧力）。

          F_int = (|F_left_y| + |F_right_y|) / 2

        内力与合力正交：
          合力 = F_left_y + F_right_y  （两者方向相反时 ≈ 0，工件不平动）
          内力 = (|F_L| + |F_R|) / 2  （夹紧应力，工件内部承受）
        """
        f_l = self._contact_force_y('left')
        f_r = self._contact_force_y('right')
        return (abs(f_l) + abs(f_r)) / 2.0