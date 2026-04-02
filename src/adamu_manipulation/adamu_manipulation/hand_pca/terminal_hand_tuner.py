import threading

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from adamu_manipulation.simple_hand_controller import (
    HAND_CLOSE,
    HAND_EDGE_GRASP,
    HAND_OPEN,
    SimpleHandController,
)


HELP_TEXT = """
可用命令：
  help                         显示帮助
  show                         显示当前 12 维关节角（rad）
  open                         发送张手预设
  close                        发送闭手预设
  edge                         发送边棱抓取预设
  set <idx> <val>              设置第 idx 个关节为 val（idx: 0~11）
  inc <idx> <delta>            第 idx 个关节增量调整 delta
  setall v1 ... v12            一次设置 12 个关节
  save                         打印当前数组（可复制到代码中）
  quit / exit                  退出
"""
#1.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.2000, 0.0000, 0.1500, 0.0000, 0.1500, 0.0000, 0.1000

def _fmt(arr: np.ndarray) -> str:
    return '[' + ', '.join(f'{x:.4f}' for x in arr.tolist()) + ']'


def main(args=None):
    rclpy.init(args=args)

    side = 'right'
    print('请选择手: left / right（默认 right）')
    raw_side = input('side> ').strip().lower()
    if raw_side in ('left', 'right'):
        side = raw_side

    hand = SimpleHandController(side=side, node_name=f'{side}_terminal_hand_tuner')
    executor = MultiThreadedExecutor()
    executor.add_node(hand)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    joints = HAND_OPEN.copy()
    hand.set_joints_immediate(joints)

    print(f'\n✅ 终端手指调参器已启动（side={side}）')
    print(HELP_TEXT)

    try:
        while True:
            cmdline = input('hand> ').strip()
            if not cmdline:
                continue

            tokens = cmdline.split()
            cmd = tokens[0].lower()

            if cmd in ('quit', 'exit'):
                break
            if cmd == 'help':
                print(HELP_TEXT)
                continue
            if cmd == 'show':
                print(_fmt(joints))
                continue
            if cmd == 'open':
                joints = HAND_OPEN.copy()
                hand.set_joints_immediate(joints)
                print('已发送 open')
                continue
            if cmd == 'close':
                joints = HAND_CLOSE.copy()
                hand.set_joints_immediate(joints)
                print('已发送 close')
                continue
            if cmd == 'edge':
                joints = HAND_EDGE_GRASP.copy()
                hand.set_joints_immediate(joints)
                print('已发送 edge')
                continue
            if cmd == 'save':
                print('可复制数组：')
                print(_fmt(joints))
                continue

            if cmd == 'set':
                if len(tokens) != 3:
                    print('用法: set <idx> <val>')
                    continue
                try:
                    idx = int(tokens[1])
                    val = float(tokens[2])
                except ValueError:
                    print('参数必须是数字')
                    continue
                if idx < 0 or idx >= 12:
                    print('idx 必须在 0~11')
                    continue
                joints[idx] = val
                hand.set_joints_immediate(joints)
                print(f'joint[{idx}] = {val:.4f}')
                continue

            if cmd == 'inc':
                if len(tokens) != 3:
                    print('用法: inc <idx> <delta>')
                    continue
                try:
                    idx = int(tokens[1])
                    delta = float(tokens[2])
                except ValueError:
                    print('参数必须是数字')
                    continue
                if idx < 0 or idx >= 12:
                    print('idx 必须在 0~11')
                    continue
                joints[idx] += delta
                hand.set_joints_immediate(joints)
                print(f'joint[{idx}] += {delta:.4f} -> {joints[idx]:.4f}')
                continue

            if cmd == 'setall':
                if len(tokens) != 13:
                    print('用法: setall v1 v2 ... v12')
                    continue
                try:
                    vals = np.array([float(x) for x in tokens[1:]], dtype=float)
                except ValueError:
                    print('所有值都必须是数字')
                    continue
                joints = vals
                hand.set_joints_immediate(joints)
                print('已发送 12 维关节')
                continue

            print('未知命令，输入 help 查看用法')

    except KeyboardInterrupt:
        print('\n收到 Ctrl+C，正在退出...')
    finally:
        executor.shutdown()
        hand.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
