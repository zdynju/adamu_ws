#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3D
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np

# TF2 依赖
from tf2_ros import Buffer, TransformListener, TransformException, StaticTransformBroadcaster
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, TransformStamped

class ColorVision3DNode(Node):
    def __init__(self):
        super().__init__('color_vision_3d_node')
        self.bridge = CvBridge()

        # ================= 1. 初始化目标颜色范围 (HSV) =================
        self.lower_red1 = np.array([0,   100,  20])
        self.upper_red1 = np.array([10,  255, 255])
        self.lower_red2 = np.array([170, 100,  20])
        self.upper_red2 = np.array([180, 255, 255])

        self.get_logger().info("✅ 视觉节点已启动：支持动态高度估计 (Ring Sampling)")

        # ================= 2. 初始化 TF2 监听器 =================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._publish_optical_frame_tf()

        # ================= 3. 相机内参 =================
        self.fx = 623.538
        self.fy = 623.538
        self.cx = 640.0
        self.cy = 360.0

        # ================= 4. 深度图参数 =================
        self.depth_is_normalized = False   
        self.depth_near = 0.01             
        self.depth_far  = 10.0             

        # ================= 5. 发布者与订阅者 =================
        self.target_pub = self.create_publisher(Detection3D, '/vision/grasp_target', 10)

        self.rgb_sub   = message_filters.Subscriber(self, Image, '/overhead_cam/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/overhead_cam/aligned_depth_to_color/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

    def _publish_optical_frame_tf(self):
        broadcaster = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp     = self.get_clock().now().to_msg()
        tf.header.frame_id  = 'overhead_cam'
        tf.child_frame_id   = 'overhead_cam_optical'
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = 1.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 0.0
        broadcaster.sendTransform(tf)

    def sync_callback(self, rgb_msg, depth_msg):
        try:
            cv_rgb   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

            if self.depth_is_normalized:
                cv_depth = self.depth_near + cv_depth * (self.depth_far - self.depth_near)

            hsv   = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask  = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((5, 5), np.uint8)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                area    = cv2.contourArea(max_cnt)

                if area > 800:
                    rect = cv2.minAreaRect(max_cnt)
                    box  = cv2.boxPoints(rect)
                    box  = box.astype(np.intp)

                    M        = cv2.moments(max_cnt)
                    if M['m00'] == 0:
                        return
                    u_center = int(M['m10'] / M['m00'])
                    v_center = int(M['m01'] / M['m00'])

                    # ================= 高鲁棒性深度提取 (箱子顶面) =================
                    obj_mask = np.zeros_like(cv_depth, dtype=np.uint8)
                    cv2.drawContours(obj_mask, [box], 0, 255, -1)
                    obj_mask_eroded = cv2.erode(obj_mask, np.ones((7,7), np.uint8), iterations=1)
                    
                    valid_depths = cv_depth[(obj_mask_eroded == 255) & (~np.isnan(cv_depth)) & (cv_depth > 0)]
                    
                    if len(valid_depths) == 0:
                        self.get_logger().warn("目标区域内未检测到有效深度", throttle_duration_sec=2.0)
                        return

                    # 箱子顶面的深度
                    Z_c = float(np.median(valid_depths))

                    # ================= 动态高度估计 (Ring Sampling) =================
                    # 制造一个紧贴物体的“外环”区域来采样桌面深度
                    kernel_ring = np.ones((30, 30), np.uint8)
                    dilated_mask = cv2.dilate(obj_mask, kernel_ring, iterations=1)
                    ring_mask = cv2.bitwise_xor(dilated_mask, obj_mask) # 抠出纯空心环
                    
                    # 提取环形区域内的深度（即桌面的深度，距离相机更远）
                    valid_table_depths = cv_depth[(ring_mask == 255) & (~np.isnan(cv_depth)) & (cv_depth > 0)]
                    
                    if len(valid_table_depths) > 100:
                        Z_table = float(np.median(valid_table_depths))
                        # 物理高度 = 桌面深度 - 箱顶深度
                        physical_height = Z_table - Z_c
                    else:
                        physical_height = 0.20 # 如果看不到桌面，使用默认高度兜底
                        self.get_logger().warn("未检测到桌面深度，使用默认高度 0.2m", throttle_duration_sec=2.0)
                        
                    # 过滤传感器噪声，强制限定高度在合理范围 (例如 2cm 到 40cm)
                    physical_height = float(np.clip(physical_height, 0.02, 0.40))

                    # ================= 计算真实长宽 + Z 轴旋转角 =================
                    (_, _), (w_pix, h_pix), angle = rect
                    physical_w = (w_pix * Z_c) / self.fx
                    physical_h = (h_pix * Z_c) / self.fy

                    length = float(max(physical_w, physical_h))
                    width  = float(min(physical_w, physical_h))

                    # ── 短边法向量（即双臂抓取时的接近方向）──────────────────────
                    # minAreaRect: angle ∈ (-90, 0]，width 边方向 = angle 度（相对图像 +X）
                    #   width >= height → width 是长边 → 短边法向 = angle + 90°
                    #   width <  height → width 是短边 → 短边法向 = angle°
                    angle_rad = np.deg2rad(angle)
                    if w_pix >= h_pix:
                        approach_angle_img = angle_rad + np.pi / 2.0
                    else:
                        approach_angle_img = angle_rad

                    # 单位方向向量（相机光学坐标系 XY 平面，Z=0）
                    dir_x_cam = float(np.cos(approach_angle_img))
                    dir_y_cam = float(np.sin(approach_angle_img))

                    # 反投影到相机光学坐标系
                    X_c = (u_center - self.cx) * Z_c / self.fx
                    Y_c = (v_center - self.cy) * Z_c / self.fy

                    world_point = self.transform_to_world(X_c, Y_c, Z_c)
                    if world_point is None:
                        return

                    # 将方向向量同样变换到世界坐标系，求世界系 yaw
                    grasp_yaw_world = self._compute_grasp_yaw(
                        X_c, Y_c, Z_c, dir_x_cam, dir_y_cam
                    )

                    # ── 利用箱子 180° 对称性，取绝对值最小的等价角 ──────────
                    # 例：顺时针 100° → 等价逆时针 80°，取 -80°（更小）
                    grasp_yaw_world = (grasp_yaw_world + np.pi / 2) % np.pi - np.pi / 2

                    # ─ 用四元数（绕世界 Z 轴旋转 yaw）编码抓取朝向 ──────────────
                    qz = float(np.sin(grasp_yaw_world / 2.0))
                    qw = float(np.cos(grasp_yaw_world / 2.0))

                    self.publish_target(
                        world_point.x, world_point.y, world_point.z,
                        length, width, physical_height,
                        qz=qz, qw=qw
                    )

                    # ── 可视化 ─────────────────────────────────────────────────
                    cv2.drawContours(cv_rgb, [box], 0, (0, 255, 0), 2)
                    cv2.circle(cv_rgb, (u_center, v_center), 5, (0, 0, 255), -1)

                    # 短边法向箭头（蓝色）
                    arrow_len = 60
                    ax = int(u_center + dir_x_cam * arrow_len)
                    ay = int(v_center + dir_y_cam * arrow_len)
                    cv2.arrowedLine(cv_rgb, (u_center, v_center), (ax, ay),
                                    (255, 100, 0), 2, tipLength=0.3)

                    # 原始 minAreaRect angle（灰色小字，调试用）
                    raw_deg = float(np.rad2deg(approach_angle_img))
                    cv2.putText(cv_rgb, f"raw_angle:{angle:.1f}  approach:{raw_deg:.1f}",
                                (max(0, u_center - 120), max(15, v_center - 45)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

                    yaw_deg = float(np.rad2deg(grasp_yaw_world))
                    tx = max(0, u_center - 120)
                    cv2.putText(cv_rgb, f"X:{world_point.x:.3f}  Y:{world_point.y:.3f}  Z:{world_point.z:.3f}",
                                (tx, max(20, v_center - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(cv_rgb, f"L:{length:.3f}  W:{width:.3f}  H:{physical_height:.3f}",
                                (tx, max(20, v_center - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(cv_rgb, f"Yaw:{yaw_deg:.1f} deg",
                                (tx, max(20, v_center + 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)

            cv2.imshow("Detection Result", cv_rgb)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"处理异常: {e}")

    def _compute_grasp_yaw(self, xc, yc, zc,
                            dir_x_cam: float, dir_y_cam: float) -> float:
        """
        将相机光学系中的方向向量变换到世界系，返回世界系 XY 平面内的 yaw 角（弧度）。
        通过变换中心点和偏移点两个 PointStamped 来利用已有 TF 链。
        """
        STEP = 0.05   # 偏移步长（米），足够大以减小数值误差
        p0 = self.transform_to_world(xc, yc, zc)
        p1 = self.transform_to_world(xc + dir_x_cam * STEP,
                                     yc + dir_y_cam * STEP,
                                     zc)
        if p0 is None or p1 is None:
            return 0.0
        return float(np.arctan2(p1.y - p0.y, p1.x - p0.x))

    def transform_to_world(self, x_c, y_c, z_c):
        point_cam = PointStamped()
        point_cam.header.frame_id = 'overhead_cam_optical'
        point_cam.header.stamp    = self.get_clock().now().to_msg()
        point_cam.point.x = x_c
        point_cam.point.y = y_c
        point_cam.point.z = z_c
        try:
            transform = self.tf_buffer.lookup_transform(
                'world', 'overhead_cam_optical',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return tf2_geometry_msgs.do_transform_point(point_cam, transform).point
        except TransformException as ex:
            self.get_logger().warn(f"TF 变换失败: {ex}", throttle_duration_sec=2.0)
            return None

    def publish_target(self, x_top, y_top, z_top, length, width, height,
                       qz: float = 0.0, qw: float = 1.0):
        """发布目标位姿和尺寸。bbox.center.orientation 编码绕世界 Z 轴的抓取 yaw。"""
        msg = Detection3D()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        msg.bbox.center.position.x = float(x_top)
        msg.bbox.center.position.y = float(y_top)
        # Z 从顶面向下移到物体中心
        msg.bbox.center.position.z = float(z_top - height / 2.0)

        # 用四元数编码抓取方向（绕世界 Z 轴的 yaw，短边法向方向）
        msg.bbox.center.orientation.x = 0.0
        msg.bbox.center.orientation.y = 0.0
        msg.bbox.center.orientation.z = float(qz)
        msg.bbox.center.orientation.w = float(qw)

        msg.bbox.size.x = length
        msg.bbox.size.y = width
        msg.bbox.size.z = height

        self.target_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ColorVision3DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()