"""
pick.py — 双臂夹取并搬起箱子

流程
----
  1. 预抓取：两臂移动到箱子两侧的预接触位置
  2. 探触：两臂慢速向中心推进，力传感器检测接触
  3. 夹紧：小步继续推进，直到内力达到目标值
  4. 抬升：两臂同步向上移动

设计原则
--------
全程使用 arm_controller 的轨迹控制器，不使用 Servo。

原因：夹持状态下 TCP 被箱子挡住，Servo 内部 J⁺ 重解会导致
零空间泄漏引起手腕旋转。轨迹控制器直接维持目标关节位置，
位置误差产生接触力，没有 IK 重解，没有零空间泄漏。

力的产生方式
------------
轨迹控制器持续尝试到达目标关节位置。TCP 被箱子挡住到不了目标，
位置误差积累，关节控制器的刚度把这个误差转化成接触力。
步长越小，力的波动越小。
"""

import asyncio
from dataclasses import dataclass

from geometry_msgs.msg import Pose
from adamu_manipulation.arm_controller import AdamuDualArmController
from adamu_manipulation.fts_processor   import FTSProcessor


# ─────────────────────────────────────────────────────────────────────────────
#  参数
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PickParams:
    """
    夹取和搬起的所有参数。

    调参顺序
    --------
    1. contact_force_threshold：先调大（10N），确认不误触发，再逐步降低
    2. search_step：从 5mm 开始，确认接触稳定后可以适当增大
    3. target_grasp_force：从小往大，确认箱子不会滑落
    4. buildup_step：越小力越稳定，建议不超过 1mm
    5. lift_height：根据实际任务设定
    """

    # ── 探触 ──────────────────────────────────────────────────────────────────

    # 每步推进距离（m），建议 0.003~0.010
    # 太大：可能冲过接触点；太小：探触时间长
    search_step: float = 0.005

    # 最大探触距离（m），防止找不到箱子一直推
    search_max_distance: float = 1

    # 接触检测阈值（N），Y 方向力超过此值认为已接触
    # 太小：传感器噪声误触发；太大：已经压进去了
    contact_force_threshold: float = 5.0

    # ── 夹紧 ──────────────────────────────────────────────────────────────────

    # 每步夹紧距离（m），建议 0.0005~0.001
    # 步长越小，力的波动越小，但夹紧过程越慢
    buildup_step: float = 0.01

    # 目标夹紧内力（N）
    # 估算：F > box_mass * 9.8 / (2 * friction_coeff)
    # 例：2kg 箱子，mu=0.4 → F > 2*9.8/(2*0.4) = 24.5N
    target_grasp_force: float = 20.0

    # 过载保护（N），超过此值立即停止
    max_grasp_force: float = 40.0

    # 夹紧最大步数，防止无限推进
    buildup_max_steps: int = 100

    # ── 抬升 ──────────────────────────────────────────────────────────────────

    # 抬升高度（m）
    lift_height: float = 0.10


# ─────────────────────────────────────────────────────────────────────────────
#  夹取搬起主体
# ─────────────────────────────────────────────────────────────────────────────

class PickPhase:
    """
    双臂夹取并搬起箱子。

    调用前提
    --------
    - arm_controller 的所有服务已就绪（wait_for_services 已调用）
    - fts_processor 已完成 tare（在预抓取位置完成零偏标定）
    - 箱子位置已知，pregrasp_left / pregrasp_right 已计算好
    """

    def __init__(
        self,
        arm:    AdamuDualArmController,
        fts:    FTSProcessor,
        params: PickParams = None,
    ):
        self.arm    = arm
        self.fts    = fts
        self.params = params or PickParams()

    # ── 对外接口 ─────────────────────────────────────────────────────────────

    async def run(
        self,
        pregrasp_left:  Pose,   # 左臂预抓取位姿（箱子左侧外 20~30mm）
        pregrasp_right: Pose,   # 右臂预抓取位姿（箱子右侧外 20~30mm）
    ) -> bool:
        """
        执行完整夹取搬起序列。

        返回 True：箱子已抬离地面，可以进入下一阶段。
        返回 False：失败，上层处理（退回预抓取位置或报警）。
        """
        logger = self.arm.get_logger()
        logger.info('=== Pick 开始 ===')

        # ── 阶段 1：预抓取位置 ────────────────────────────────────────────
        logger.info('[1/4] 移动到预抓取位置...')
        ok = await self._move_to_pregrasp(pregrasp_left, pregrasp_right)
        if not ok:
            logger.error('[1/4] 移动到预抓取位置失败')
            return False

        # ── 阶段 2：探触 ──────────────────────────────────────────────────
        logger.info('[2/4] 开始探触...')
        ok = await self._search_contact()
        if not ok:
            logger.error('[2/4] 探触失败，未检测到接触')
            return False

        # ── 阶段 3：夹紧 ──────────────────────────────────────────────────
        logger.info('[3/4] 开始夹紧...')
        ok = await self._buildup_grasp()
        if not ok:
            logger.error('[3/4] 夹紧失败')
            return False

        # ── 阶段 4：抬升 ──────────────────────────────────────────────────
        logger.info('[4/4] 开始抬升...')
        ok = await self._lift()
        if not ok:
            logger.error('[4/4] 抬升失败')
            # 抬升失败时已经夹住了，缓慢放回原位再松开
            await self._lower_back()
            return False

        logger.info('=== Pick 完成 ===')
        return True

    # ── 阶段实现 ─────────────────────────────────────────────────────────────

    async def _move_to_pregrasp(
        self,
        left_pose:  Pose,
        right_pose: Pose,
    ) -> bool:
        """
        两臂并发移动到预抓取位置。

        使用关节空间规划（RRT），避免碰撞。
        两臂独立规划，此时没有接触约束，不需要联合规划。
        """
        left_ok, right_ok = await asyncio.gather(
            self.arm.send_single_arm_goal('left',  left_pose),
            self.arm.send_single_arm_goal('right', right_pose),
        )
        return left_ok and right_ok

    async def _search_contact(self) -> bool:
        """
        探触阶段。

        两臂同时以小步向中心推进，每步执行完检查力传感器。
        检测到接触后停止。

        左臂向右（+Y），右臂向左（-Y）。
        每步用 execute_single_arm_straight_line，
        TCP 碰到箱子后轨迹控制器产生位置误差 → 接触力。
        没有 Servo，没有零空间泄漏。
        """
        p      = self.params
        logger = self.arm.get_logger()

        contacted_left  = False
        contacted_right = False
        total           = 0.0

        while total < p.search_max_distance:

            # 检查两侧的力
            if not contacted_left:
                f_l = self._force_y('left')
                if f_l is not None and abs(f_l) >= p.contact_force_threshold:
                    contacted_left = True
                    logger.info(f'左臂接触确认 F_y={f_l:.2f}N')

            if not contacted_right:
                f_r = self._force_y('right')
                if f_r is not None and abs(f_r) >= p.contact_force_threshold:
                    contacted_right = True
                    logger.info(f'右臂接触确认 F_y={f_r:.2f}N')

            # 双侧都接触，结束探触
            if contacted_left and contacted_right:
                logger.info(f'双臂接触完成，已推进 {total*1000:.1f}mm')
                return True

            # 构造本步的推进方向
            # 已接触的一侧不再推进（delta = 0），未接触的继续推进
            left_dy  = -p.search_step if not contacted_left  else 0.0
            right_dy = +p.search_step if not contacted_right else 0.0

            # 两侧都还没接触：并发推进
            # 一侧已接触：只推进未接触的一侧
            tasks = []
            if left_dy != 0.0:
                tasks.append(
                    self.arm.execute_single_arm_straight_line(
                        'left', (0.0, left_dy, 0.0)
                    )
                )
            if right_dy != 0.0:
                tasks.append(
                    self.arm.execute_single_arm_straight_line(
                        'right', (0.0, right_dy, 0.0)
                    )
                )

            if tasks:
                results = await asyncio.gather(*tasks)
                if not all(results):
                    logger.error('探触推进规划失败')
                    return False

            total += p.search_step

        # 超出最大距离仍未双侧接触
        logger.warn(
            f'探触超出最大距离 {p.search_max_distance*1000:.0f}mm，'
            f'左={contacted_left} 右={contacted_right}'
        )
        # 只要有一侧接触就继续，两侧都没接触才算失败
        return contacted_left or contacted_right

    async def _buildup_grasp(self) -> bool:
        """
        夹紧阶段。

        接触已建立，继续小步推进直到内力达到目标值。

        步长比探触更小（0.5~1mm），给力传感器足够的响应时间，
        防止内力超调压坏箱子。

        力的来源：轨迹控制器试图到达目标关节位置，
        箱子阻止 TCP 移动，位置误差 × 关节刚度 = 接触力。
        不需要 Servo，不需要导纳。
        """
        p      = self.params
        logger = self.arm.get_logger()

        for step in range(p.buildup_max_steps):

            f_int = self._internal_force()

            if f_int is None:
                logger.warn('力传感器无数据，跳过本步')
                continue

            # 过载保护
            if f_int > p.max_grasp_force:
                logger.error(f'过载 {f_int:.1f}N > {p.max_grasp_force}N，停止夹紧')
                return False

            # 达到目标
            if f_int >= p.target_grasp_force:
                logger.info(f'夹紧力达标 {f_int:.1f}N（目标 {p.target_grasp_force}N）')
                return True

            logger.info(
                f'[夹紧 {step+1}/{p.buildup_max_steps}] '
                f'内力={f_int:.1f}N，继续推进 {p.buildup_step*1000:.1f}mm'
            )

            # 两臂同步向中心推进一小步
            left_ok, right_ok = await asyncio.gather(
                self.arm.execute_single_arm_straight_line(
                    'left',  (0.0, -p.buildup_step, 0.0)
                ),
                self.arm.execute_single_arm_straight_line(
                    'right', (0.0, +p.buildup_step, 0.0)
                ),
            )

            if not (left_ok and right_ok):
                logger.error(f'夹紧推进规划失败（步 {step+1}）')
                return False

        # 超出最大步数
        f_int = self._internal_force()
        logger.warn(
            f'夹紧超出最大步数 {p.buildup_max_steps}，'
            f'当前内力={f_int:.1f}N'
        )
        # 允许 80% 达标继续
        return f_int is not None and f_int >= p.target_grasp_force * 0.8

    async def _lift(self) -> bool:
        """
        抬升阶段。

        夹紧力由轨迹控制器的位置偏差被动维持，
        两臂同步 Z 方向上移，间距几何上不变。

        为什么间距不变：
        两臂路点的 Y 方向增量都是 0，X 方向增量都是 0，
        只有 Z 方向相同的增量，所以间距向量在整个抬升过程中不变。
        """
        ok = await self.arm.execute_dual_arm_straight_line(
            left_delta  = (0.0, 0.0, +self.params.lift_height),
            right_delta = (0.0, 0.0, +self.params.lift_height),
        )
        if ok:
            self.arm.get_logger().info(
                f'抬升完成 {self.params.lift_height*1000:.0f}mm'
            )
        return ok

    async def _lower_back(self):
        """
        抬升失败时缓慢放回原位（安全退出）。
        """
        self.arm.get_logger().warn('抬升失败，放回原位...')
        await self.arm.execute_dual_arm_straight_line(
            left_delta  = (0.0, 0.0, -self.params.lift_height),
            right_delta = (0.0, 0.0, -self.params.lift_height),
        )

    # ── 力传感器辅助 ─────────────────────────────────────────────────────────

    def _force_y(self, side: str):
        """
        读取指定侧 Y 方向力（N）。
        fts_processor 在预抓取位置完成 tare，读到的是纯接触力。
        返回 None 表示传感器无数据。
        """
        force = self.fts.get_standardized_force(side)
        if force is None:
            return None
        return force[1]

    def _internal_force(self):
        """
        计算内力（夹紧力）。

          F_int = (|F_left_y| + |F_right_y|) / 2

        内力与合力正交：
          合力 = F_L + F_R ≈ 0（两臂方向相反，工件不平动）
          内力 = 均值（工件内部夹紧应力）

        任一侧传感器无数据时返回 None。
        """
        f_l = self._force_y('left')
        f_r = self._force_y('right')
        if f_l is None or f_r is None:
            return None
        return (abs(f_l) + abs(f_r)) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  调用示例
# ─────────────────────────────────────────────────────────────────────────────

async def main_pick_example(arm, fts, box_pose):
    """
    调用示例。

    box_pose：箱子中心的世界坐标位姿。
    pregrasp 位姿在箱子两侧法线方向偏移 30mm，高度对准箱子重心。
    """
    import copy

    # 计算预抓取位姿：在箱子两侧 Y 方向偏移 30mm
    pregrasp_left  = copy.deepcopy(box_pose)
    pregrasp_right = copy.deepcopy(box_pose)
    pregrasp_left.position.y  -= 0.03   # 左臂在箱子左侧
    pregrasp_right.position.y += 0.03   # 右臂在箱子右侧

    pick = PickPhase(
        arm=arm,
        fts=fts,
        params=PickParams(
            search_step=0.005,
            contact_force_threshold=5.0,
            target_grasp_force=20.0,
            max_grasp_force=40.0,
            buildup_step=0.001,
            lift_height=0.10,
        ),
    )

    success = await pick.run(pregrasp_left, pregrasp_right)

    if success:
        arm.get_logger().info('箱子已搬起，可以进入下一阶段')
    else:
        arm.get_logger().error('夹取失败，检查箱子位置和传感器')

    return success