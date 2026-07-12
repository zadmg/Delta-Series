import cv2
import numpy as np
from pathlib import Path

# 1. 设置目标图片路径
SMALL_MAP_PATH = "1.1.png"

# 2. 设置模板图片路径
BIG_MAP_PATH = "1.0.png"

# 3. 目标图尺寸
SIZE1 = 186 #图像长
SIZE2 = 188 #图像宽


# ================== 主程序 ==================
def main():

    # ================== 读取目标图片 ==================
    small_map = cv2.imread(SMALL_MAP_PATH)

    # 确保小地图尺寸为
    if small_map.shape[:2] != (SIZE1, SIZE2):
        #print(f"小地图尺寸不匹配 原尺寸 {small_map.shape[:2]}，将自动缩放到 {SIZE}x{SIZE}")
        small_map = cv2.resize(small_map, (SIZE1, SIZE2))

    gray = cv2.cvtColor(small_map, cv2.COLOR_BGR2GRAY)

    # ================== 读取模板图片 ==================
    big_map = cv2.imread(BIG_MAP_PATH)

    big_map_gray = cv2.cvtColor(big_map, cv2.COLOR_BGR2GRAY)

    # ================== 特征匹配 ==================
    sift = cv2.SIFT_create()
    kp_big, des_big = sift.detectAndCompute(big_map_gray, None)
    kp_query, des_query = sift.detectAndCompute(gray, None)

    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_query, des_big, k=2)

    # 筛选匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        raise ValueError("匹配点不足！请确保小地图包含清晰地形特征")

    # 计算变换矩阵
    src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_big[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 计算目标图片中心点在模板图上的位置
    center_query = np.float32([[[SIZE1 // 2, SIZE2 // 2]]])
    center_big = cv2.perspectiveTransform(center_query, M)

    # 获取坐标（转换为整数）
    x, y = map(int, center_big[0][0])

    # 保存出坐标
    with open("location.txt", "w") as f:
        f.write(f"x: {x}\ny: {y}")
    # ================== 显示结果 ==================
    print(f"定位成功！当前点在模板的坐标: ({x}, {y})")
    result = big_map.copy()
    # 计算矩形框的坐标
    box_size1 = SIZE1
    box_size2 = SIZE2
    left = max(0, x - box_size1 // 2)
    top = max(0, y - box_size2 // 2)
    right = min(result.shape[1], x + box_size1 // 2)
    bottom = min(result.shape[0], y + box_size2 // 2)

    # 绘制红色矩形框
    cv2.rectangle(result, (left, top), (right, bottom), (0, 0, 255), 2)

    # 绘制红色中心点
    cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

    # 显示坐标信息
    cv2.putText(result, f"({x}, {y})", (left + 5, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    # 保存结果图
    output_path = Path(SMALL_MAP_PATH).stem + "_result.png"
    cv2.imwrite(output_path, result)
    print(f"结果已保存: {output_path}")

    # 显示结果（可选，用于调试）
    cv2.imshow("定位结果", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()