import colour
import numpy as np


def normalize_hdr(img, bits):
    return np.float32(img) / (2 ** bits)


def pq_2_linear(PQ_values):
    """
        将PQ编码的值转换为线性光照值。

        Parameters:
        PQ_values : array
            PQ编码的图像数据或单个值。

        Returns:
        array
            线性域中的光照值。
        """
    return colour.models.eotf_ST2084(PQ_values)


def change_exposure(img, factor):
    return img * factor


def apply_gamma(img, exposure_time, gamma):
    output_image = np.clip(np.power((img * exposure_time), 1 / gamma), 0, 1)
    return output_image


def rec2020_2_sRGB(img):
    # 定义源色域和目标色域
    rec2020 = colour.models.RGB_COLOURSPACE_BT2020
    sRGB = colour.models.RGB_COLOURSPACE_sRGB
    max_luminance = img.max()
    # 规范化图像数据到 [0, 1]
    normalized_img = img / max_luminance
    # 执行色域转换
    sRGB_image = colour.RGB_to_RGB(
        normalized_img,
        rec2020,
        sRGB,
        apply_cctf_decoding=False,  # 因为图像是线性的
        apply_cctf_encoding=False  # 保持线性输出
    )
    restored_luminance_img = sRGB_image * max_luminance
    return np.clip(restored_luminance_img, 0, np.inf)


def calculate_dynamic_exposure(img, target_max=0.9):
    """
    根据图像最大亮度动态计算曝光时间。

    参数:
    - img: 输入的HDR图像数据。
    - target_max: 目标最大亮度，调整后的最大像素值应接近此值。

    返回:
    - 动态计算的曝光时间。
    """
    max_luminance = img.max()
    if max_luminance == 0:
        return 1  # 避免除以零
    return target_max / max_luminance


def mean_based_exposure(hdr_img, target_mean=0.55, gamma=2.2, tol=0.078, max_iter=100):
    # 确定曝光时间的初始范围
    low, high = 0, 1
    iter_num = 0
    mid = 0
    for _ in range(max_iter):
        mid = (low + high) / 2
        # 应用曝光时间和Gamma校正
        output_image = np.clip(np.power((hdr_img * mid), 1 / gamma), 0, 1)
        current_mean = np.mean(output_image)

        # 检查当前的平均值与目标平均值的差异
        if abs(current_mean - target_mean) < tol:
            # 如果在容差范围内，则返回当前曝光时间
            return mid
        elif current_mean < target_mean:
            # 如果平均值太低，增加曝光时间
            low = mid
        else:
            # 如果平均值太高，减少曝光时间
            high = mid
        iter_num = _
    print(f'iter:{iter_num}')
    print(f'the exposure time: {mid}')
    # 如果达到最大迭代次数仍未找到合适的曝光时间，则返回最后一次尝试的曝光时间
    return mid


def histogram_based_exposure(hdr_img, target_percentile=90, target_value=0.9, gamma=2.2, tol=0.01, max_iter=100):
    low, high = 0, 1
    mid = 0
    for _ in range(max_iter):
        mid = (low + high) / 2
        output_image = np.clip(np.power((hdr_img * mid), 1 / gamma), 0, 1)
        current_percentile = np.percentile(output_image, target_percentile)

        if abs(current_percentile - target_value) < tol:
            return mid
        elif current_percentile < target_value:
            low = mid
        else:
            high = mid
    print(f'iter:{_}')
    return mid
