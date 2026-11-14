# ---- Carter 2D 라이다 기본 코드 ----
# 앞쪽과 뒤쪽 2D 라이다는 /front_2d_lidar/scan, /back_2d_lidar/scan 토픽(sensor_msgs/LaserScan)으로 발행됩니다.
# 360도, 빔 3200개, 0.05-30 m 입니다. 약 6 Hz 로 도착하므로 가장 최근 스캔이 0.2 s 가량 늦을 수 있습니다.

# 장착 위치와 방향: 로봇 기준 (x 앞, y 왼쪽, rad). /tf 에서 잰 값입니다.
_LIDAR = {
    "/front_2d_lidar/scan": (0.026, 0.0, 3.142),
    "/back_2d_lidar/scan": (-0.49, 0.0, 3.142),
}
# 로봇 외형(로봇 기준 x 최소, x 최대, y 최소, y 최대). 이 안에 맞은 점은 로봇 자신의 몸체입니다.
_BODY = (-0.70, 0.25, -0.34, 0.34)

_node = None
_rclpy = None
_scans = {}  # 토픽 -> 가장 최근 스캔의 점 목록 (로봇 기준 (x, y))


def sensor_setup():
    """ROS 2 노드와 구독을 한 번만 만듭니다. rclpy 는 여기서 가져옵니다.
    모듈 맨 위에서 import rclpy 를 하면 Isaac Sim 이 켜지기 전이라 실행이 실패합니다."""
    global _node, _rclpy
    if _node is not None:
        return
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    _rclpy = rclpy
    _node = rclpy.create_node("b2_lidar_reader")
    for topic in _LIDAR:
        _node.create_subscription(LaserScan, topic, lambda m, k=topic: _on_scan(k, m), qos_profile_sensor_data)


def _on_scan(topic, msg):
    """스캔의 빔마다 점을 만듭니다. 빔 방향 = 장착 방향 + 빔 각도, 점 = 장착 위치 + 거리 x 방향."""
    mx, my, myaw = _LIDAR[topic]
    pts = []
    a = msg.angle_min
    for r in msg.ranges:
        if msg.range_min <= r <= msg.range_max:
            bx = mx + r * math.cos(myaw + a)
            by = my + r * math.sin(myaw + a)
            if not (_BODY[0] <= bx <= _BODY[1] and _BODY[2] <= by <= _BODY[3]):
                pts.append((bx, by))
        a += msg.angle_increment
    _scans[topic] = pts


def sensor_points():
    """두 라이다의 가장 최근 스캔을 합친 점 목록 (로봇 기준 (x 앞, y 왼쪽)). controller() 에서 매 호출마다 부르십시오.
    처음 몇 번의 호출에서는 빈 목록이 나올 수 있습니다."""
    sensor_setup()
    _rclpy.spin_once(_node, timeout_sec=0.0)
    pts = []
    for v in _scans.values():
        pts.extend(v)
    return pts


def to_world(points, pose):
    """로봇 기준 점을 월드 (x, y) 로 바꿉니다. pose 는 controller() 가 준 (x, y, yaw) 입니다."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in points]
